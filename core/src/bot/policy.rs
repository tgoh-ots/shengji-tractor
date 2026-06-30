//! Difficulty-aware bot policy.
//!
//! [`select_action`] dispatches by [`BotDifficulty`] but always derives its
//! decision ONLY from the redacted, per-player view (`GameState::for_player`).
//! See the honesty-invariant note on [`select_action`].
//!
//! The honest tiers share a single heuristic backbone ([`crate::bot::heuristics`])
//! and differ in a small set of *knobs* (and, for `Expert`, a learned net):
//!
//! | tier       | card memory  | blunder ε | softmax temp | determinized search          |
//! |------------|--------------|-----------|--------------|------------------------------|
//! | Easy       | full public  | ~6%       | warm         | no                           |
//! | Expert     | full public  | ~0%       | greedy       | learned-net prior            |
//! | Enoch      | full public  | ~0%       | greedy       | playbook heuristic           |
//! | Grandmaster| full history | 0%        | greedy       | playbook + full-hand rollout |
//! | Omniscient | (cheats)     | 0%        | greedy       | perfect-info search          |
//!
//! `Expert` scores each legal candidate with a small MLP distilled from the
//! `Omniscient` teacher's choices (see [`crate::bot::expert`]); it consumes only
//! HONEST per-candidate features and serves as the PRIOR of a time-boxed
//! determinized search, so it approximates perfect-info play from the redacted
//! view. When the model can't load/run, the search prior transparently falls
//! back to the shared hand-written heuristic.
//!
//! Whenever any tier's logic fails to produce a move, we fall back to the
//! original always-legal "dumb" policy so a bot never makes an illegal/None
//! move when it must act.

use std::time::Duration;

use anyhow::Error;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_mechanics::trick::TrickUnit;
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Rank, Trump};

use crate::bot::heuristics::{self, ScoredPlay};
use crate::bot::phase;
use crate::bot::search::{search_play, search_play_perfect_info, Policy, SearchConfig};
use crate::bot::BotDifficulty;
use crate::game_state::play_phase::PlayPhase;
use crate::game_state::GameState;
use crate::interactive::Action;

/// Per-tier behavioural knobs.
#[derive(Clone, Copy, Debug)]
struct Knobs {
    /// Probability of making a random legal "blunder" instead of a scored move.
    epsilon: f64,
    /// Softmax temperature over the top candidate moves (higher = more random
    /// among the good moves). 0 means pick the top move deterministically.
    temperature: f64,
    /// How many determinized worlds to sample in search. 0 disables search and
    /// the tier plays directly from the heuristic backbone (Easy). Expert/Enoch
    /// do a deeper, time-boxed search.
    search_worlds: usize,
    /// Maximum candidate moves the search evaluates.
    search_candidates: usize,
    /// How many tricks of look-ahead each rollout plays. Deeper = stronger but
    /// slower.
    rollout_tricks: usize,
}

impl Knobs {
    fn for_difficulty(d: BotDifficulty) -> Self {
        match d {
            // Beginner: occasional blunders, a warm (but no longer scalding)
            // softmax over the top moves, and no look-ahead search. It retains
            // public history like every honest tier, but remains
            // clearly the weakest, beatable tier — it just makes fewer obvious
            // blunders than before. The blunder rate and softmax temperature are
            // the two knobs that gate its strength; both were nudged DOWN
            // (ε 0.28→0.06, temp 3.5→1.1) so it sticks to the heuristic's top
            // suggestions more often without ever gaining real look-ahead. The
            // `easy_ab_benchmark` example measures Easy@new vs Easy@old at these
            // values (a modest ~56% bump — noticeable but not a blowout).
            BotDifficulty::Easy => Knobs {
                epsilon: 0.06,
                temperature: 1.1,
                search_worlds: 0,
                search_candidates: 0,
                rollout_tricks: 0,
            },
            // Expert: a learned net scores each legal candidate from HONEST
            // features and serves as the PRIOR of a time-boxed determinized
            // search (see `choose_play` / `crate::bot::expert`). The search knobs
            // below drive that search; if the net can't run, the search prior
            // transparently falls back to the shared hand-written heuristic, so
            // Expert is never illegal/None. ε = 0 and temperature = 0 so the
            // search value is taken greedily.
            //
            // Deepened search: the move is computed off the game lock and masked
            // by the ~1200ms visible pacing (see `search_budget_ms`), so a bigger
            // budget buys strength without lagging chat/UI. More determinized
            // worlds cut the per-decision value variance and a deeper rollout
            // sharpens the leaf estimate. In practice these positions finish well
            // under the 2.2s cap (avg ~tens of ms), so the worlds/rollout-depth
            // bumps — not the wall-clock budget — are what deepen the search; the
            // budget is the safety ceiling.
            BotDifficulty::Expert => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                search_worlds: 144,
                search_candidates: 6,
                rollout_tricks: 12,
            },
            // Enoch: the SAME time-boxed determinized search as Expert, but driven
            // by the Enoch-playbook heuristic (`Policy::EnochHeuristic`) at both
            // the root prior and the rollout plies, so the full-game strategy
            // shapes the search rather than only a one-shot greedy pick. No
            // blunders, greedy on the search value. Honest (own redacted view).
            BotDifficulty::Enoch => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                // Same deepened determinized search as Expert, driven by the
                // Enoch-playbook heuristic at both the prior and the rollout plies.
                search_worlds: 144,
                search_candidates: 6,
                rollout_tricks: 12,
            },
            // Grandmaster: the apex HONEST tier. Same determinized-search
            // machinery and the Enoch-playbook policy (so it inherits Enoch's
            // perfect play-memory determinization), but with a WIDER root (more
            // candidates so it rarely prunes the right move), a higher world cap,
            // full-hand rollouts (`GM_ROLLOUT=0`), and a larger budget
            // (`GM_BUDGET_MULT`). Knobs are env-overridable (`GM_WORLDS` /
            // `GM_CANDS` / `GM_ROLLOUT`) so the self-play harness can sweep the
            // search shape without recompiling; the defaults below are the tuned
            // production values. ε = 0, temperature = 0 (greedy on the search
            // value).
            BotDifficulty::Grandmaster => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                // Tuned defaults (env-overridable for sweeps): a WIDE root (8
                // candidates), a high world cap (400) so a large budget is fully
                // used, and `GM_ROLLOUT=0` = roll every world out to the LAST card
                // (exact terminal points — no truncation bias). Self-play showed the
                // full-rollout shape ties Enoch at equal budget but converts a
                // larger search budget into a clear win, whereas Enoch (capped at
                // 144 worlds / 12-trick rollouts) plateaus. See `GM_BUDGET_MULT`.
                search_worlds: env_usize("GM_WORLDS", 400),
                search_candidates: env_usize("GM_CANDS", 8),
                rollout_tricks: env_usize("GM_ROLLOUT", 0),
            },
            // CHEATER (perfect information): plays a perfect-information search
            // over the SINGLE true world (no determinization, no sampling — it
            // already knows every opponent's cards via the centralized honesty
            // bypass). ε = 0 and temperature = 0 (purely greedy on the search
            // value), all candidates considered, and deep/full rollouts since
            // there's only one world to evaluate. `search_worlds` here is reused
            // as "rollouts per candidate" by `search_play_perfect_info`.
            BotDifficulty::Omniscient => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                // `search_worlds` here is reused as "rollouts per candidate" by
                // `search_play_perfect_info`. With the larger off-lock budget
                // (`OMNI_BUDGET_MULT`, up to ~15s) we average MANY full-hand
                // rollouts per candidate, driving down the variance from the
                // rollouts' exploration noise so the greedy pick over the TRUE
                // world is reliable. Each rollout is a full playout; the budget
                // still caps the work. Env-overridable (`OMNI_WORLDS`).
                search_worlds: env_usize("OMNI_WORLDS", 32),
                search_candidates: usize::MAX,
                rollout_tricks: usize::MAX,
            },
        }
    }
}

/// Select the next legal [`Action`] for the bot identified by `me`, given ONLY
/// the redacted, per-player view of the game state.
///
/// # Honesty invariant
///
/// `view` MUST be the redacted view obtained from
/// `GameState::for_player(me)` / `InteractiveGame::dump_state_for_player(me)`.
/// In that view every other seat's cards are replaced with [`Card::Unknown`] and
/// the kitty is hidden, so this function structurally cannot read information a
/// human in `me`'s seat couldn't. We never accept the unredacted state here. The
/// determinized search likewise only *samples* the hidden cards; it never reads
/// them.
pub fn select_action(
    view: &GameState,
    me: PlayerID,
    difficulty: BotDifficulty,
) -> Result<Option<Action>, Error> {
    select_action_impl(view, me, difficulty, None)
}

/// Training/evaluation entry point with an explicit FINAL play-search budget.
/// Unlike the process-wide environment knob, this is per call: Omniscient does
/// not apply its production multiplier, and concurrent behavior/Q policies can
/// use independent budgets without mutating global state.
pub fn select_action_with_search_budget(
    view: &GameState,
    me: PlayerID,
    difficulty: BotDifficulty,
    search_budget_ms: u64,
) -> Result<Option<Action>, Error> {
    select_action_impl(view, me, difficulty, Some(search_budget_ms.max(1)))
}

fn select_action_impl(
    view: &GameState,
    me: PlayerID,
    difficulty: BotDifficulty,
    search_budget_override_ms: Option<u64>,
) -> Result<Option<Action>, Error> {
    match view {
        // Lobby configuration is a human concern.
        GameState::Initialize(_) => Ok(None),
        GameState::Draw(p) => {
            if p.done_drawing() {
                Ok(None)
            } else if p.next_player()? == me {
                Ok(Some(Action::DrawCard))
            } else {
                Ok(None)
            }
        }
        GameState::Exchange(p) => exchange_action(p, me, difficulty),
        GameState::Play(p) => {
            if p.game_finished() {
                return Ok(None);
            }
            match p.trick().next_player() {
                Some(next) if next == me => {
                    let cards = choose_play(p, me, difficulty, search_budget_override_ms);
                    Ok(Some(Action::PlayCards(cards)))
                }
                _ => Ok(None),
            }
        }
    }
}

/// The search-tier wall-clock budget in milliseconds. Defaults to 2200ms;
/// overridable via the `SHENGJI_BOT_BUDGET_MS` environment variable so the
/// self-play eval harness (and the test suite) can trade strength for speed in
/// bulk runs.
///
/// # Why 2200ms is safe in production
///
/// The backend computes the bot's move on a `spawn_blocking` worker that holds
/// NEITHER the game lock NOR an async runtime thread (see
/// `drive_bots_non_blocking` in `backend/src/shengji_handler.rs`), so the search
/// runs entirely concurrently with chat / other players' actions. The bot's
/// VISIBLE move is then paced by `DEFAULT_BOT_ACTION_PAUSE_MS` (~1200ms), which
/// overlaps the search rather than adding to it. A ~2.2s search therefore masks
/// under the pacing and keeps the total bot turn comfortably under ~3s while
/// giving the determinized search markedly more worlds/depth than the old 1000ms.
/// Tests override this to a few ms, so the suite stays fast.
fn search_budget_ms() -> u64 {
    std::env::var("SHENGJI_BOT_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(2200)
}

/// Read a `usize` tuning knob from the environment, falling back to `default`
/// when the variable is unset or unparseable. Used by the Grandmaster tier so the
/// self-play harness can sweep its search shape without recompiling.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Read an `f64` tuning knob from the environment, falling back to `default` when
/// unset/unparseable.
fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// Read a [`Policy`] tuning knob from the environment (`heuristic`/`net`/`enoch`,
/// or `0`/`1`/`2`), falling back to `default`. Lets the self-play harness sweep
/// Grandmaster's prior / rollout policy without recompiling.
fn env_policy(name: &str, default: Policy) -> Policy {
    match std::env::var(name).ok().as_deref() {
        Some("heuristic") | Some("0") => Policy::Heuristic,
        Some("net") | Some("1") => Policy::Net,
        Some("enoch") | Some("2") => Policy::EnochHeuristic,
        _ => default,
    }
}

fn seed_bytes(state: &mut u64, bytes: &[u8]) {
    // FNV-1a is intentionally simple and stable across processes/toolchains;
    // unlike HashMap iteration or RandomState, it gives repeatable eval seeds.
    for byte in bytes {
        *state ^= u64::from(*byte);
        *state = state.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn seed_u64(state: &mut u64, value: u64) {
    seed_bytes(state, &value.to_le_bytes());
}

fn seed_cards(state: &mut u64, cards: impl IntoIterator<Item = Card>) {
    let cards: Vec<Card> = cards.into_iter().collect();
    seed_u64(state, cards.len() as u64);
    for card in cards {
        seed_u64(state, card.as_char() as u32 as u64);
    }
}

/// A canonical hash of exactly what an honest player can observe. It includes
/// card identities/counts, attributed public history, score/role, trump, and
/// public configuration; it deliberately never walks another seat's hand or a
/// hidden kitty, even when called with Omniscient's unredacted state.
pub(crate) fn observation_seed(p: &PlayPhase, me: PlayerID) -> u64 {
    let mut seed = 0xCBF2_9CE4_8422_2325u64;
    seed_u64(&mut seed, me.0 as u64);
    seed_u64(&mut seed, p.num_decks() as u64);
    seed_bytes(&mut seed, format!("{:?}", p.trump()).as_bytes());

    // Own hand only, in mechanics-canonical order.
    if let Ok(hand) = p.hands().get(me) {
        let mut entries: Vec<(Card, usize)> =
            hand.iter().map(|(&card, &count)| (card, count)).collect();
        entries.sort_by(|(a, _), (b, _)| {
            p.trump()
                .compare(*a, *b)
                .then_with(|| a.as_char().cmp(&b.as_char()))
        });
        seed_u64(&mut seed, entries.len() as u64);
        for (card, count) in entries {
            seed_u64(&mut seed, card.as_char() as u32 as u64);
            seed_u64(&mut seed, count as u64);
        }
    }

    // Completed history retains trick boundaries and seat attribution.
    seed_u64(&mut seed, p.public_play_history().len() as u64);
    for completed in p.public_play_history() {
        seed_u64(&mut seed, completed.len() as u64);
        for played in completed {
            seed_u64(&mut seed, played.id.0 as u64);
            seed_cards(&mut seed, played.cards.iter().copied());
            seed_cards(&mut seed, played.bad_throw_cards.iter().copied());
            seed_u64(
                &mut seed,
                played.better_player.map(|id| id.0 as u64 + 1).unwrap_or(0),
            );
        }
    }

    // Old persisted states may have only the aggregate history, so include its
    // canonical multiset too. Current trick order/attribution is hashed directly.
    let mut played_counts: Vec<(Card, usize)> = p
        .played_this_hand()
        .iter()
        .map(|(&card, &count)| (card, count))
        .collect();
    played_counts.sort_by(|(a, _), (b, _)| {
        p.trump()
            .compare(*a, *b)
            .then_with(|| a.as_char().cmp(&b.as_char()))
    });
    for (card, count) in played_counts {
        seed_u64(&mut seed, card.as_char() as u32 as u64);
        seed_u64(&mut seed, count as u64);
    }
    for played in p.trick().played_cards() {
        seed_u64(&mut seed, played.id.0 as u64);
        seed_cards(&mut seed, played.cards.iter().copied());
        seed_cards(&mut seed, played.bad_throw_cards.iter().copied());
    }
    for player in p.trick().player_queue() {
        seed_u64(&mut seed, player.0 as u64);
    }
    seed_u64(
        &mut seed,
        p.trick()
            .winner_so_far()
            .map(|id| id.0 as u64 + 1)
            .unwrap_or(0),
    );

    // Hard public voids, score orientation, role, and all public rule settings.
    let mut voids: Vec<_> = p.voids_this_hand().iter().collect();
    voids.sort_by_key(|(player, _)| player.0);
    for (player, suits) in voids {
        seed_u64(&mut seed, player.0 as u64);
        let mut suits = suits.clone();
        suits.sort_unstable();
        seed_bytes(&mut seed, format!("{:?}", suits).as_bytes());
    }
    let (non_landlord_points, observed_points) = p.calculate_points();
    seed_u64(&mut seed, non_landlord_points as i64 as u64);
    seed_u64(&mut seed, observed_points as i64 as u64);
    seed_u64(&mut seed, u64::from(p.landlords_team().contains(&me)));

    // Hash public gameplay configuration, but scrub room metadata which cannot
    // affect a card decision.  Serialising PropagatedState wholesale used to
    // make the bot's random choices change when somebody joined as an observer,
    // renamed a player, changed the chat link, or edited the bot registry.
    if let Ok(mut config) = serde_json::to_value(p.propagated()) {
        if let Some(object) = config.as_object_mut() {
            for key in [
                "observers",
                "max_player_id",
                "num_games_finished",
                "landlord_emoji",
                "chat_link",
                "special_decks",
                "bots",
            ] {
                object.remove(key);
            }
            if let Some(players) = object.get_mut("players").and_then(|v| v.as_array_mut()) {
                for player in players {
                    if let Some(player) = player.as_object_mut() {
                        player.remove("name");
                    }
                }
            }
        }
        if let Ok(config) = serde_json::to_vec(&config) {
            seed_bytes(&mut seed, &config);
        }
    }

    // Special-deck ordering is not semantic. Hash the exact configured card
    // multiset instead, so equivalent deck configurations receive one seed.
    if let Some(mut configured) = p.configured_cards_for_determinization() {
        configured.sort_by_key(|card| card.as_char());
        seed_cards(&mut seed, configured);
    }

    // Public seat order and remaining hand sizes matter, but hidden identities
    // never do. This is identical for full and per-player-redacted states.
    for player in p.propagated().players() {
        seed_u64(&mut seed, player.id.0 as u64);
        let hand_size = p
            .hands()
            .get(player.id)
            .map(|hand| hand.values().sum::<usize>())
            .unwrap_or(0);
        seed_u64(&mut seed, hand_size as u64);
    }
    let mut team = p.landlords_team().to_vec();
    team.sort_by_key(|id| id.0);
    for id in team {
        seed_u64(&mut seed, id.0 as u64);
    }
    seed_u64(&mut seed, p.landlord().0 as u64);
    seed_u64(&mut seed, p.exchanger().0 as u64);
    for bid in p.public_bids() {
        seed_u64(&mut seed, bid.id.0 as u64);
        seed_u64(&mut seed, bid.card.as_char() as u32 as u64);
        seed_u64(&mut seed, bid.count as u64);
        seed_u64(&mut seed, bid.epoch as u64);
    }

    let (kitty, removed) = p.piles_for_determinization();
    seed_u64(&mut seed, kitty.len() as u64);
    // Only the exchanger observed the buried identities. Full-state callers do
    // not get to smuggle them into an honest tier's seed.
    if me == p.exchanger() {
        let mut kitty = kitty.to_vec();
        kitty.sort_by_key(|card| card.as_char());
        seed_cards(&mut seed, kitty);
    }
    let mut removed = removed.to_vec();
    removed.sort_by_key(|card| card.as_char());
    seed_cards(&mut seed, removed);

    // Friend declarations are a set, but they originate in a HashSet. Sort a
    // canonical representation so insertion/random iteration order is ignored.
    match p.game_mode() {
        crate::settings::GameMode::Tractor => seed_u64(&mut seed, 0),
        crate::settings::GameMode::FindingFriends {
            num_friends,
            friends,
        } => {
            seed_u64(&mut seed, 1);
            seed_u64(&mut seed, *num_friends as u64);
            let mut friends = friends.clone();
            friends.sort_by_key(|friend| {
                (
                    friend.card.as_char(),
                    friend.initial_skip,
                    friend.skip,
                    friend.player_id.map(|id| id.0),
                )
            });
            for friend in friends {
                seed_u64(&mut seed, friend.card.as_char() as u32 as u64);
                seed_u64(&mut seed, friend.initial_skip as u64);
                seed_u64(&mut seed, friend.skip as u64);
                seed_u64(
                    &mut seed,
                    friend.player_id.map(|id| id.0 as u64 + 1).unwrap_or(0),
                );
            }
        }
    }

    // SplitMix64 finalizer improves diffusion for nearby observations.
    seed ^= seed >> 30;
    seed = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    seed ^= seed >> 27;
    seed = seed.wrapping_mul(0x94D0_49BB_1331_11EB);
    seed ^ (seed >> 31)
}

/// Seed an RNG deterministically from the canonical honest observation.
fn rng_for(p: &PlayPhase, me: PlayerID) -> StdRng {
    StdRng::seed_from_u64(observation_seed(p, me))
}

/// Choose the cards to play in the current trick for the given difficulty.
/// Always returns a legal play (falling back to the dumb policy on any failure).
fn choose_play(
    p: &PlayPhase,
    me: PlayerID,
    difficulty: BotDifficulty,
    search_budget_override_ms: Option<u64>,
) -> Vec<Card> {
    let knobs = Knobs::for_difficulty(difficulty);
    let mut rng = rng_for(p, me);
    let leading = p.trick().played_cards().is_empty();

    // ε-greedy blunder: occasionally make a random legal move (beginner feel).
    if rng.gen_bool(knobs.epsilon.clamp(0.0, 1.0)) {
        if let Some(cards) = random_legal_play(p, me, &mut rng, leading) {
            return cards;
        }
    }

    // CHEATER tier: Omniscient runs a PERFECT-INFORMATION search. The `p`
    // (`PlayPhase`) it was handed already contains the REAL hands for every seat,
    // because the driver obtained it through the centralized honesty bypass
    // (`crate::bot::observed_state`, the sole intentional perfect-information
    // path). So instead of sampling hidden hands like the honest search tiers, we
    // search the single true world directly with full rollouts — see
    // `search_play_perfect_info`.
    if matches!(difficulty, BotDifficulty::Omniscient) {
        // Omniscient sees the TRUE world, so to actually EXPLOIT that it must
        // search with the strongest available POLICY — the Enoch full-game
        // playbook — not the bare heuristic. (With the plain heuristic it was
        // observed to LOSE to the playbook-driven Enoch even with perfect
        // information: better strategy beat better information.) The playbook
        // shapes both the candidate prior and the full-hand rollouts over the real
        // hands, so each candidate is scored by its exact terminal outcome when
        // every seat plays well. The cheater is also allowed to think the longest
        // of any tier: `OMNI_BUDGET_MULT` (default 5×) scales its budget, capped at
        // ~15s, run OFF the game lock by the non-blocking driver.
        let omni_budget_ms = search_budget_override_ms.unwrap_or_else(|| {
            ((search_budget_ms() as f64 * env_f64("OMNI_BUDGET_MULT", 5.0)) as u64).min(14_500)
        });
        let config = SearchConfig {
            time_budget: Duration::from_millis(omni_budget_ms),
            max_worlds: knobs.search_worlds.max(1),
            max_candidates: knobs.search_candidates,
            rollout_tricks: knobs.rollout_tricks,
            seed: rng.gen(),
            // Prior + rollout policy are env-selectable for sweeps; default to the
            // Enoch playbook prior (best candidate ranker on the true world).
            policy: env_policy("OMNI_PRIOR", Policy::EnochHeuristic),
            rollout_policy: env_policy("OMNI_ROLLOUT_POLICY", Policy::EnochHeuristic),
        };
        if let Some(cards) = search_play_perfect_info(p, me, config) {
            return cards;
        }
    }

    // Expert / Enoch: time-boxed determinized search over sampled worlds. Both
    // tiers share the SAME search machinery (determinizer + world sampling +
    // rollouts + the static leaf evaluator); the ONLY difference is the POLICY
    // that supplies the candidate prior (root pruning) and shapes the rollouts:
    //
    //   * Expert → `Policy::Net` PRIOR (the distilled learned net ranks/prunes the
    //              root candidates) + heuristic rollouts. The net is a far better
    //              *root prior* than it is a cheap deep-rollout policy, so we use
    //              it where it pays off (choosing which candidates to search) and
    //              keep the fast heuristic for the many rollout plies. This is
    //              AlphaZero-lite: search + a learned policy prior, static leaf
    //              value unchanged. If the net can't run, the prior transparently
    //              falls back to the hand-written heuristic, so Expert is never
    //              illegal/None.
    //   * Enoch  → `Policy::EnochHeuristic` prior + Enoch-playbook rollouts (the
    //              full-game competitive playbook shapes both plies).
    //
    // The wall-clock budget defaults to 2200ms but can be lowered via
    // `SHENGJI_BOT_BUDGET_MS` (used by the self-play eval harness and the test
    // suite to keep large runs fast). The search runs OFF the game lock and is
    // masked by the bot's ~1200ms visible pacing, so the larger budget buys
    // strength without lagging chat/UI. The cap still guarantees a slow CPU
    // degrades to fewer simulations rather than hanging. Both honest tiers read
    // ONLY the redacted view `p`.
    if knobs.search_worlds > 0 {
        // Enoch shapes BOTH the prior AND the rollout plies with its full-game
        // playbook; Expert uses the learned-net prior (which itself falls back to
        // the bare hand-written heuristic if the net can't run). Grandmaster
        // reuses the Enoch proposal policy but searches deeper: full-hand
        // rollouts, a wider candidate root, more worlds, and a larger budget.
        let enoch = matches!(difficulty, BotDifficulty::Enoch);
        let grandmaster = matches!(difficulty, BotDifficulty::Grandmaster);
        // Grandmaster deliberately plays a DIFFERENT style from Enoch at equal
        // strength. Its prior and rollout policy are independently selectable
        // (`GM_PRIOR` / `GM_ROLLOUT_POLICY`: heuristic | net | enoch). The chosen
        // identity is: Enoch-playbook PRIOR (so it proposes the same sensible
        // candidate moves — no naked-joker opens, tractor-first, etc.) but NEUTRAL
        // plain-heuristic ROLLOUTS for the leaf value. The
        // effect: Enoch greedily obeys its hand-coded defensive playbook, whereas
        // Grandmaster commits to whatever its full-hand SIMULATIONS value highest —
        // a calculation-driven player that will break the playbook's instincts when
        // the deep rollout disagrees. Self-play (n=1200, paired): statistically
        // TIED with Enoch on win-rate (~50–52%) — equal strength, different
        // decisions. The Enoch prior remains the measured stronger proposal set.
        let prior = if matches!(difficulty, BotDifficulty::Expert) {
            Policy::Net
        } else if grandmaster {
            env_policy("GM_PRIOR", Policy::EnochHeuristic)
        } else if enoch {
            Policy::EnochHeuristic
        } else {
            Policy::Heuristic
        };
        let rollout_policy = if grandmaster {
            env_policy("GM_ROLLOUT_POLICY", Policy::Heuristic)
        } else if enoch {
            Policy::EnochHeuristic
        } else {
            Policy::Heuristic
        };
        // Grandmaster: `GM_ROLLOUT=0` (its default) means "roll every sampled world
        // out to the LAST card" (exact terminal points, no truncation bias) — the
        // single biggest honest lever over Enoch's 12-trick truncated rollouts.
        let rollout_tricks = if grandmaster && knobs.rollout_tricks == 0 {
            usize::MAX
        } else {
            knobs.rollout_tricks
        };
        // Grandmaster is the apex tier and is allowed to *think longer* than the
        // other tiers (a higher difficulty searches more worlds / deeper). This is
        // honest — pure extra computation on its own redacted view, never extra
        // information. `GM_BUDGET_MULT` scales its per-decision wall-clock budget;
        // the default of 3× is what converts its full-hand-rollout search into a
        // clear win over Enoch in self-play (Enoch, capped at 144 worlds / 12-trick
        // rollouts, plateaus and cannot use the extra time). At the production base
        // (2200ms) that is ~6.6s/decision, run OFF the game lock by the
        // non-blocking bot driver and masked by the visible pacing, so it never
        // lags chat/UI; lower `GM_BUDGET_MULT` (or `SHENGJI_BOT_BUDGET_MS`) to trade
        // strength for speed.
        let budget_ms = search_budget_override_ms.unwrap_or_else(|| {
            if grandmaster {
                (search_budget_ms() as f64 * env_f64("GM_BUDGET_MULT", 3.0)) as u64
            } else {
                search_budget_ms()
            }
        });
        let config = SearchConfig {
            time_budget: Duration::from_millis(budget_ms),
            max_worlds: knobs.search_worlds,
            max_candidates: knobs.search_candidates.max(1),
            rollout_tricks,
            seed: rng.gen(),
            policy: prior,
            // Rollouts use the cheap heuristic default policy (see the
            // `rollout_policy` doc on `SearchConfig`) — Enoch keeps the playbook in
            // its rollouts; Grandmaster deliberately rolls out with the NEUTRAL
            // heuristic (calculation-driven, decoupled from the playbook's
            // defensive instincts — see the prior/rollout selection above).
            rollout_policy,
        };
        if let Some(cards) = search_play(p, me, config) {
            return cards;
        }
    }

    // Heuristic backbone (Easy, and the search tiers' fallback). Every tier at
    // or above Enoch inherits the playbook-enabled greedy fallback.
    let enoch = matches!(
        difficulty,
        BotDifficulty::Enoch | BotDifficulty::Grandmaster | BotDifficulty::Omniscient
    );
    let ranked: Vec<ScoredPlay> = if leading {
        if enoch {
            heuristics::ranked_leads_enoch(p, me)
        } else {
            heuristics::ranked_leads(p, me)
        }
    } else if enoch {
        heuristics::ranked_follows_enoch(p, me)
    } else {
        heuristics::ranked_follows(p, me)
    };
    if let Some(cards) = pick_from_ranked(&ranked, knobs.temperature, &mut rng) {
        return cards;
    }

    // Final always-legal fallback: the original dumb policy.
    dumb_play(p, me).unwrap_or_default()
}

/// Pick a candidate from a heuristic-ranked list using softmax sampling with the
/// given temperature. A temperature of 0 picks the top move deterministically.
fn pick_from_ranked(
    ranked: &[ScoredPlay],
    temperature: f64,
    rng: &mut StdRng,
) -> Option<Vec<Card>> {
    if ranked.is_empty() {
        return None;
    }
    if temperature <= 0.0 {
        return Some(ranked[0].cards.clone());
    }
    // Restrict to the top handful so the softmax stays focused on good moves.
    let top = &ranked[..ranked.len().min(4)];
    let max = top
        .iter()
        .map(|s| s.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = top
        .iter()
        .map(|s| ((s.score - max) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Some(ranked[0].cards.clone());
    }
    let mut roll = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        roll -= w;
        if roll <= 0.0 {
            return Some(top[i].cards.clone());
        }
    }
    Some(top[0].cards.clone())
}

/// Pick a uniformly-random legal play (used for ε-blunders).
fn random_legal_play(
    p: &PlayPhase,
    me: PlayerID,
    rng: &mut StdRng,
    leading: bool,
) -> Option<Vec<Card>> {
    use rand::seq::SliceRandom;
    let mut candidates = if leading {
        heuristics::lead_candidates(p, me)
    } else {
        heuristics::follow_candidates(p, me)
    };
    candidates.shuffle(rng);
    candidates.into_iter().next()
}

// ===========================================================================
// Exchange phase (kitty + bidding-side decisions)
// ===========================================================================

fn exchange_action(
    p: &crate::game_state::exchange_phase::ExchangePhase,
    me: PlayerID,
    difficulty: BotDifficulty,
) -> Result<Option<Action>, Error> {
    if p.next_player()? != me {
        return Ok(None);
    }

    // With kitty theft enabled, a successful overbid makes a non-landlord the
    // exchanger. That seat must bury/finalize just like the original landlord.
    // Once an exchange is finalized, only the landlord performs friend selection
    // and starts play; overbid/pickup resolution is coordinated by the driver.
    let arranging_kitty = !p.finalized();
    if !arranging_kitty && p.landlord() != me {
        return Ok(None);
    }

    let trump = p.trump();

    // Determine how many cards we still need to bury. The kitty already holds
    // the correct number when the phase begins; if we want to bury *different*
    // cards we must first move some out and move our chosen ones in. Simpler and
    // robust: figure out our chosen burial set and reconcile one card at a time.
    let hand = p.hands().get(me).ok();
    if arranging_kitty {
        if let Some(hand) = hand {
            let hand_cards: Vec<Card> = Card::cards(hand.iter()).copied().collect();
            // The kitty size equals however many cards are currently buried; we keep
            // it constant. We compute a desired burial from the *hand* and swap.
            // Count current kitty contents we can see (we're the exchanger, so the
            // kitty is visible to us in the unredacted-for-exchanger view).
            // The redacted view hides the kitty unless we're the exchanger, which we
            // are here, so it should be visible. We compute desired buries from the
            // combined hand+kitty pool to make the best choice.
            // To keep within the validated API, we only ever swap one card per call:
            // move a sub-optimal kitty card to hand, or move a good-to-bury hand card
            // to the kitty. The driver calls us repeatedly until we BeginPlay.
            if let Some(action) = reconcile_kitty(p, me, &hand_cards, trump, difficulty) {
                return Ok(Some(action));
            }
        }

        // Theft mode has an explicit finalization/overbid round. Starting play
        // directly here is illegal; put the kitty down and let the driver offer
        // legal overbids (or let the landlord start when nobody bids).
        if p.kitty_theft_enabled() {
            return Ok(Some(Action::PutDownKitty));
        }
    }

    // Friends (FindingFriends only). Do this once: repeatedly sending SetFriends
    // prevented a bot landlord from ever reaching BeginPlay.
    let num_friends = p.num_friends();
    if num_friends > 0 && !p.friends_selected() {
        let mut viable = p.valid_friend_selections();
        viable.sort_by_key(|friend| {
            std::cmp::Reverse((
                trump.effective_suit(friend.card) != EffectiveSuit::Trump,
                heuristics::card_strength(trump, friend.card),
                std::cmp::Reverse(friend.initial_skip),
            ))
        });
        viable.truncate(num_friends);
        if viable.len() == num_friends {
            return Ok(Some(Action::SetFriends(viable)));
        }
    }

    Ok(Some(Action::BeginPlay))
}

/// Decide a single kitty-reconciliation step that converges on burying the
/// globally-worst `kitty_size` cards from the combined hand+kitty pool, while
/// preserving the honesty boundary (the kitty is visible to the exchanger).
///
/// The exchange API only moves one card at a time and only checks the kitty
/// size at `BeginPlay`, so we converge in two strictly-decreasing phases:
///
/// 1. **Evict**: if the kitty holds a card that is NOT in our desired burial
///    multiset, pull it into the hand. (This shrinks the wrong-card count.)
/// 2. **Bury**: otherwise, if a desired-burial card is still in the hand, move
///    it to the kitty.
///
/// When the kitty exactly equals the desired burial set, we return `None` and
/// the caller proceeds to `BeginPlay`. If the kitty is hidden (shouldn't happen
/// for the exchanger) we conservatively do nothing so we never act on garbage.
fn reconcile_kitty(
    p: &crate::game_state::exchange_phase::ExchangePhase,
    _me: PlayerID,
    hand_cards: &[Card],
    trump: Trump,
    difficulty: BotDifficulty,
) -> Option<Action> {
    let kitty_size = p.kitty_size();
    if kitty_size == 0 {
        return None;
    }
    let kitty = p.visible_kitty()?;

    // Desired burial = the worst `kitty_size` cards of the COMBINED pool, chosen
    // deterministically (see `choose_kitty`, which breaks ties by a stable card
    // ordering). Because the pool (hand ∪ kitty) is invariant under moves, this
    // target is STABLE across calls, which is what makes the reconciliation
    // terminate. Enoch applies its stricter, point-budgeted burial discipline.
    let mut pool: Vec<Card> = hand_cards.to_vec();
    pool.extend_from_slice(kitty);
    let advanced = matches!(
        difficulty,
        BotDifficulty::Enoch | BotDifficulty::Grandmaster | BotDifficulty::Omniscient
    );
    // The current learned kitty contract is trained against the plain
    // heuristic target. Grandmaster uses Enoch's structurally different
    // baseline, so applying that artifact there would be out-of-distribution.
    let desired =
        if difficulty == BotDifficulty::Expert && phase::kitty_domain_supported(p, pool.len()) {
            phase::choose_kitty(&pool, trump, kitty_size, advanced)
        } else if advanced {
            heuristics::choose_kitty_enoch(&pool, trump, kitty_size)
        } else {
            heuristics::choose_kitty(&pool, trump, kitty_size)
        };

    // Compute the symmetric difference as multisets: which desired cards are
    // still missing from the kitty (to bury), and which kitty cards are not in
    // desired (to evict). We act with a SINGLE rule so we never undo our own
    // previous move: if the kitty is under-full, BURY a missing card; only if
    // the kitty is at (or over) the target size do we EVICT a wrong card. This
    // ordering guarantees strict progress toward `desired` and cannot oscillate.
    let desired_counts = Card::count(desired.iter().copied());
    let kitty_counts = Card::count(kitty.iter().copied());

    // Cards missing from the kitty that we'd like to bury (in desired, short in
    // kitty), restricted to those actually held in the hand.
    let hand_counts = Card::count(hand_cards.iter().copied());
    let mut missing: Vec<Card> = vec![];
    for (&card, &want) in &desired_counts {
        let have = kitty_counts.get(&card).copied().unwrap_or(0);
        let in_hand = hand_counts.get(&card).copied().unwrap_or(0);
        let short = want.saturating_sub(have).min(in_hand);
        for _ in 0..short {
            missing.push(card);
        }
    }

    // Kitty cards that aren't part of the desired burial (wrong cards to evict).
    let mut wrong: Vec<Card> = vec![];
    for (&card, &have) in &kitty_counts {
        let want = desired_counts.get(&card).copied().unwrap_or(0);
        let extra = have.saturating_sub(want);
        for _ in 0..extra {
            wrong.push(card);
        }
    }

    // Already reconciled: kitty is exactly the desired burial.
    if missing.is_empty() && wrong.is_empty() {
        return None;
    }

    if kitty.len() < kitty_size {
        // Under-full: bury the least-valuable missing card to fill the kitty.
        if let Some(&card) = missing.iter().min_by_key(|c| keep_value(trump, **c)) {
            return Some(Action::MoveCardToKitty(card));
        }
        // No missing card is in hand (can't improve): evict a wrong card so the
        // pool reshuffles toward the target on the next pass.
        if let Some(&card) = wrong.iter().max_by_key(|c| keep_value(trump, **c)) {
            return Some(Action::MoveCardToHand(card));
        }
        None
    } else {
        // Kitty is full (or over): evict the most valuable wrong card so we can
        // bury a better one next pass.
        if let Some(&card) = wrong.iter().max_by_key(|c| keep_value(trump, **c)) {
            return Some(Action::MoveCardToHand(card));
        }
        None
    }
}

/// How much we want to KEEP a card (not bury it). Higher = keep. Points and
/// trumps and high cards are valuable to keep.
fn keep_value(trump: Trump, card: Card) -> i32 {
    let mut v = 0;
    if card.points().is_some() {
        v += 100;
    }
    if trump.effective_suit(card) == EffectiveSuit::Trump {
        v += 60;
    }
    // Use the live Ace-high ordering. `Number::as_u32()` is a serialization-ish
    // value where Ace is 1, which previously made the reconciliation logic evict
    // Aces as if they were low trash.
    v += heuristics::card_strength(trump, card).clamp(0, 20);
    v
}

// ===========================================================================
// Bidding (used by the driver via valid_bids; exposed for completeness)
// ===========================================================================

/// Choose the best bid for `me` from the legal bids, or `None` to pass. Encodes
/// "don't overbid a weak hand": only bids when the hand has a genuinely strong
/// trump holding. The driver decides *whether* a bid is required; this picks the
/// best one when the bot chooses to bid.
pub fn choose_bid(
    p: &crate::game_state::draw_phase::DrawPhase,
    me: PlayerID,
    difficulty: BotDifficulty,
) -> Option<shengji_mechanics::bidding::Bid> {
    let valid = p.valid_bids(me).ok()?;
    if valid.is_empty() {
        return None;
    }
    // The trump number a declaration establishes is the LANDLORD's rank when a
    // landlord is pinned (every round after the first — all seats bid at the
    // landlord's level, mirroring `Bid::valid_bids` / `DrawPhase::advance`),
    // otherwise the bidder's own rank (round one, no landlord). Using `me`'s rank
    // unconditionally mis-scored the candidate trump (it counted the wrong
    // trump-number cards) once a landlord was pinned, now that bots declare in
    // those rounds too.
    let level_seat = p.propagated().landlord().unwrap_or(me);
    let level = p
        .propagated()
        .players()
        .iter()
        .find(|pl| pl.id == level_seat)
        .map(|pl| pl.rank())
        .unwrap_or(Rank::Number(Number::Two));
    let hand: Vec<Card> = p
        .hands()
        .get(me)
        .ok()
        .map(|h| Card::cards(h.iter()).copied().collect())
        .unwrap_or_default();

    // Enoch and every higher tier inherit the pair-aware playbook: a trump pair
    // is worth roughly 3-4 single trumps. Lower tiers use the simpler
    // length-/strength-weighted backbone.
    let enoch = matches!(
        difficulty,
        BotDifficulty::Enoch | BotDifficulty::Grandmaster | BotDifficulty::Omniscient
    );

    // Score each candidate bid by the trump it would establish.
    // Keep the absolute heuristic score separate from the optional model rank:
    // listwise logits have arbitrary offset/scale and must never move the
    // bid-versus-pass threshold.
    let mut evaluated = Vec::with_capacity(valid.len());
    for bid in valid {
        let candidate_trump = match bid.card {
            Card::SmallJoker | Card::BigJoker => heuristics::trump_for(level, None),
            Card::Suited { suit, .. } => heuristics::trump_for(level, Some(suit)),
            Card::Unknown => continue,
        };
        let mut strength = if enoch {
            heuristics::bid_strength_enoch(&hand, candidate_trump)
        } else {
            heuristics::bid_strength(&hand, candidate_trump)
        };
        // Prefer fewer cards committed for the same strength (reinforce later).
        strength -= bid.count as f64 * 0.5;
        evaluated.push((bid, candidate_trump, strength));
    }
    if evaluated.is_empty() {
        return None;
    }
    let heuristic_best_strength = evaluated
        .iter()
        .map(|(_, _, strength)| *strength)
        .reduce(f64::max)
        .unwrap_or(f64::NEG_INFINITY);
    // The current bid exporter covers fully-dealt Expert-style states only.
    // During-draw bids and Enoch/Grandmaster playbook scores remain heuristic
    // until matching training support exists.
    let rankings = if difficulty == BotDifficulty::Expert && phase::bid_domain_supported(p) {
        phase::rank_bid_candidates(p, me, &evaluated)
    } else {
        evaluated.iter().map(|(_, _, strength)| *strength).collect()
    };
    let mut best_index = 0usize;
    let mut best_rank = f64::NEG_INFINITY;
    for (index, rank) in rankings.into_iter().enumerate() {
        if rank > best_rank {
            best_rank = rank;
            best_index = index;
        }
    }
    let best = Some(evaluated[best_index]);

    // Advanced-tier "don't declare too early": before most of the hand has been dealt you
    // can't tell how long the suit will be, so a premature declaration "can make
    // or break your game." Require Enoch to have drawn most of its cards before it
    // commits to a trump, UNLESS its holding is already overwhelming.
    if enoch {
        let drawn = hand.len();
        // Final per-player hand size = (all cards − kitty) / players.
        let total_cards = p.cards_in_play();
        let players = p.propagated().players().len().max(1);
        let kitty = p.kitty().len();
        let full_hand = total_cards.saturating_sub(kitty) / players;
        let fraction = if full_hand == 0 {
            1.0
        } else {
            drawn as f64 / full_hand as f64
        };
        // "Overwhelming" enough to declare BEFORE the deal completes requires
        // genuine PAIR STRUCTURE (the playbook values a trump pair at ~3-4 singles),
        // not merely a high length-driven strength score — a long-but-unpaired suit
        // that happens to reach the old ~22 threshold must still wait for the rest
        // of the deal, exactly the "declared too early / wrong structure" failure
        // the playbook warns about. Demand >=2 trump pairs (or a trump tractor) AND
        // a still-high strength.
        let overwhelming = best
            .map(|(bid, _, strength)| {
                if strength < 18.0 {
                    return false;
                }
                let candidate_trump = match bid.card {
                    Card::SmallJoker | Card::BigJoker => heuristics::trump_for(level, None),
                    Card::Suited { suit, .. } => heuristics::trump_for(level, Some(suit)),
                    Card::Unknown => return false,
                };
                let (pairs, tractor) = heuristics::trump_pair_structure(&hand, candidate_trump);
                pairs >= 2 || tractor
            })
            .unwrap_or(false);
        if fraction < 0.6 && !overwhelming {
            return None;
        }
    }

    best.and_then(|(bid, _, _)| {
        // The model may choose among legal bid candidates, but it must not
        // alter the separately calibrated heuristic bid-versus-pass decision.
        if heuristic_best_strength >= 10.0 {
            Some(bid)
        } else {
            None
        }
    })
}

/// Choose a legal overbid during a kitty-theft exchange round. This mirrors the
/// draw-phase strength model but operates on the exchange phase's current epoch;
/// it is intentionally conservative so a bot steals only with a genuinely good
/// resulting trump holding.
pub fn choose_exchange_bid(
    p: &crate::game_state::exchange_phase::ExchangePhase,
    me: PlayerID,
    difficulty: BotDifficulty,
) -> Option<shengji_mechanics::bidding::Bid> {
    let valid = p.valid_bids(me).ok()?;
    if valid.is_empty() {
        return None;
    }
    let level = p
        .trump()
        .number()
        .map(Rank::Number)
        .unwrap_or(Rank::NoTrump);
    let hand: Vec<Card> = p
        .hands()
        .get(me)
        .ok()
        .map(|h| Card::cards(h.iter()).copied().collect())
        .unwrap_or_default();
    let advanced = matches!(
        difficulty,
        BotDifficulty::Enoch | BotDifficulty::Grandmaster | BotDifficulty::Omniscient
    );

    valid
        .into_iter()
        .filter_map(|bid| {
            let candidate_trump = match bid.card {
                Card::SmallJoker | Card::BigJoker => heuristics::trump_for(level, None),
                Card::Suited { suit, .. } => heuristics::trump_for(level, Some(suit)),
                Card::Unknown => return None,
            };
            let mut strength = if advanced {
                heuristics::bid_strength_enoch(&hand, candidate_trump)
            } else {
                heuristics::bid_strength(&hand, candidate_trump)
            };
            strength -= bid.count as f64 * 0.5;
            (strength >= 10.0).then_some((strength, bid))
        })
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, bid)| bid)
}

// ===========================================================================
// Always-legal fallback (the original dumb-but-legal policy)
// ===========================================================================

/// The original always-legal play used as a last-resort fallback so the bot
/// never produces an illegal / empty move when it must act.
fn dumb_play(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    let leading = p.trick().played_cards().is_empty();
    let proposed = if leading {
        dumb_lead(p, me)
    } else {
        dumb_follow(p, me)
    };
    if proposed
        .as_ref()
        .is_some_and(|cards| p.can_play_cards(me, cards).is_ok())
    {
        return proposed;
    }

    // This path should be vanishingly rare. Reuse the mechanics-validated
    // generator as the final guard so unusual tuple/bomb settings cannot turn a
    // fallback into an illegal bot action.
    let candidates = if leading {
        heuristics::lead_candidates(p, me)
    } else {
        heuristics::follow_candidates(p, me)
    };
    candidates
        .into_iter()
        .find(|cards| p.can_play_cards(me, cards).is_ok())
}

fn dumb_lead(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    use std::collections::HashMap;
    let hand = p.hands().get(me).ok()?;
    let cards: Vec<Card> = Card::cards(hand.iter()).copied().collect();
    let trump = p.trick().trump();

    let mut cards_by_suit: HashMap<EffectiveSuit, Vec<Card>> = HashMap::new();
    for card in cards {
        cards_by_suit
            .entry(trump.effective_suit(card))
            .or_default()
            .push(card);
    }

    let mut best_play: Option<Vec<Card>> = None;
    for (_, suit_cards) in cards_by_suit.into_iter() {
        let results = TrickUnit::find_plays(trump, p.propagated().tractor_requirements, suit_cards);
        let play = results
            .into_iter()
            .filter_map(|play| play.into_iter().max_by_key(|u| u.size()))
            .max_by_key(|u| u.size());
        if let Some(play) = play {
            let play_cards = play.cards();
            match &best_play {
                None => best_play = Some(play_cards),
                Some(b) if play_cards.len() > b.len() => best_play = Some(play_cards),
                Some(_) => (),
            }
        }
    }
    best_play
}

fn dumb_follow(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    use shengji_mechanics::ordered_card::OrderedCard;
    use shengji_mechanics::trick::UnitLike;

    let hand = p.hands().get(me).ok()?.clone();
    let trick_format = p.trick().trick_format()?.clone();

    let available_cards: Vec<Card> = Card::cards(
        hand.iter()
            .filter(|(c, _)| trick_format.trump().effective_suit(**c) == trick_format.suit()),
    )
    .copied()
    .collect();

    let matching_play = trick_format
        .decomposition(p.propagated().trick_draw_policy())
        .filter_map(|format| {
            let mut playable = UnitLike::check_play(
                OrderedCard::make_map(available_cards.iter().copied(), trick_format.trump()),
                format.iter().cloned(),
                p.propagated().trick_draw_policy(),
            );
            playable.next().map(|u| {
                u.into_iter()
                    .flat_map(|x| {
                        x.into_iter()
                            .flat_map(|(card, count)| std::iter::repeat_n(card.card, count))
                    })
                    .collect::<Vec<_>>()
            })
        })
        .next();

    let num_required = trick_format.size();
    let mut play = match matching_play {
        Some(matching) if matching.len() == num_required => matching,
        Some(_) if num_required >= available_cards.len() => available_cards.clone(),
        Some(mut matching) => {
            let mut remaining = available_cards.clone();
            for m in &matching {
                if let Some(pos) = remaining.iter().position(|c| *c == *m) {
                    remaining.remove(pos);
                }
            }
            let needed = num_required - matching.len();
            matching.extend(remaining.into_iter().take(needed));
            matching
        }
        None => available_cards.clone(),
    };

    let required_other_cards = num_required.saturating_sub(play.len());
    if required_other_cards > 0 {
        let other_cards: Vec<Card> = Card::cards(
            hand.iter()
                .filter(|(c, _)| trick_format.trump().effective_suit(**c) != trick_format.suit()),
        )
        .copied()
        .collect();
        play.extend(other_cards.into_iter().take(required_other_cards));
    }

    Some(play)
}
