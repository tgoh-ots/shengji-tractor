//! Difficulty-aware bot policy.
//!
//! [`select_action`] dispatches by [`BotDifficulty`] but always derives its
//! decision ONLY from the redacted, per-player view (`GameState::for_player`).
//! See the honesty-invariant note on [`select_action`].
//!
//! The honest tiers share a single heuristic backbone ([`crate::bot::heuristics`])
//! and differ in a small set of *knobs* (and, for `Expert`, a learned net):
//!
//! | tier      | card memory | blunder ε | softmax temp | determinized search    |
//! |-----------|-------------|-----------|--------------|------------------------|
//! | Easy      | none        | ~28%      | high (hot)   | no                     |
//! | Hard      | yes (voids) | ~0%       | ~greedy      | yes (time-boxed)       |
//! | Expert    | yes (voids) | ~0%       | greedy       | learned net (honest)   |
//! | Omniscient| (cheats)    | 0%        | greedy       | perfect-info search    |
//!
//! `Expert` scores each legal candidate with a small MLP distilled from the
//! `Omniscient` teacher's choices (see [`crate::bot::expert`]); it consumes only
//! HONEST per-candidate features, so it approximates perfect-info play from the
//! redacted view. When the model can't load/run, Expert falls back to the `Hard`
//! determinized search, which itself falls back to the heuristic.
//!
//! Whenever any tier's logic fails to produce a move, we fall back to the
//! original always-legal "dumb" policy so a bot never makes an illegal/None
//! move when it must act.

use std::time::Duration;

use anyhow::Error;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_mechanics::trick::{TractorRequirements, TrickUnit};
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Rank, Suit, Trump};

use crate::bot::heuristics::{self, ScoredPlay};
use crate::bot::search::{search_play, search_play_perfect_info, SearchConfig};
use crate::bot::BotDifficulty;
use crate::game_state::play_phase::PlayPhase;
use crate::game_state::GameState;
use crate::interactive::Action;
use crate::settings::FriendSelection;

/// Per-tier behavioural knobs.
#[derive(Clone, Copy, Debug)]
struct Knobs {
    /// Probability of making a random legal "blunder" instead of a scored move.
    epsilon: f64,
    /// Softmax temperature over the top candidate moves (higher = more random
    /// among the good moves). 0 means pick the top move deterministically.
    temperature: f64,
    /// How many determinized worlds to sample in search. 0 disables search and
    /// the tier plays directly from the heuristic backbone (Easy). Hard does a
    /// deeper, time-boxed search.
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
            // Beginner: frequent blunders, hot softmax over the top moves, and
            // NO card memory / search. Feels like a casual human.
            BotDifficulty::Easy => Knobs {
                epsilon: 0.28,
                temperature: 3.5,
                search_worlds: 0,
                search_candidates: 0,
                rollout_tricks: 0,
            },
            // Hard: the heuristic PLUS a time-boxed determinized search that
            // looks ahead over sampled worlds, essentially no blunders, greedy on
            // the search value. Strictly stronger than the bare heuristic when
            // the budget allows any simulations at all.
            BotDifficulty::Hard => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                search_worlds: 48,
                search_candidates: 6,
                rollout_tricks: 8,
            },
            // Expert: a learned net scores each legal candidate from HONEST
            // features (see `choose_play` / `crate::bot::expert`). These knobs
            // only matter for the fallback path when the model can't run — in
            // that case Expert behaves like Hard (determinized search). ε = 0 and
            // temperature = 0 so the net's argmax is taken greedily.
            BotDifficulty::Expert => Knobs {
                epsilon: 0.0,
                temperature: 0.0,
                search_worlds: 48,
                search_candidates: 6,
                rollout_tricks: 8,
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
                search_worlds: 8,
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
                    let cards = choose_play(p, me, difficulty);
                    Ok(Some(Action::PlayCards(cards)))
                }
                _ => Ok(None),
            }
        }
    }
}

/// The Hard-tier search wall-clock budget in milliseconds. Defaults to 1000ms;
/// overridable via the `SHENGJI_BOT_BUDGET_MS` environment variable so the
/// self-play eval harness can trade strength for speed in bulk runs.
fn search_budget_ms() -> u64 {
    std::env::var("SHENGJI_BOT_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1000)
}

/// Seed an RNG deterministically from the (redacted) play state so that, given
/// the same observable position, a bot behaves reproducibly. We derive the seed
/// from the player id, their hand size, and the number of cards on the table —
/// all things visible in the redacted view.
fn rng_for(p: &PlayPhase, me: PlayerID) -> StdRng {
    let hand_size = p
        .hands()
        .get(me)
        .map(|h| h.values().sum::<usize>())
        .unwrap_or(0);
    let on_table = p.trick().played_cards().len();
    let seed = (me.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (hand_size as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (on_table as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    StdRng::seed_from_u64(seed)
}

/// Choose the cards to play in the current trick for the given difficulty.
/// Always returns a legal play (falling back to the dumb policy on any failure).
fn choose_play(p: &PlayPhase, me: PlayerID, difficulty: BotDifficulty) -> Vec<Card> {
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
    // path). So instead of sampling hidden hands like Hard, we search the single
    // true world directly with full rollouts — see `search_play_perfect_info`.
    if matches!(difficulty, BotDifficulty::Omniscient) {
        let config = SearchConfig {
            time_budget: Duration::from_millis(search_budget_ms()),
            max_worlds: knobs.search_worlds.max(1),
            max_candidates: knobs.search_candidates,
            rollout_tricks: knobs.rollout_tricks,
            seed: rng.gen(),
        };
        if let Some(cards) = search_play_perfect_info(p, me, config) {
            return cards;
        }
    }

    // Expert (learned net): score every legal candidate with the distilled MLP
    // over HONEST per-candidate features and take the best. This is the sole
    // perfect-info-approximating path; it reads ONLY the redacted view `p`. If
    // the model can't run (failed to load, no candidates, etc.) we fall through
    // to the Hard determinized search below, so Expert is never illegal/None.
    if matches!(difficulty, BotDifficulty::Expert) {
        if let Some(cards) = crate::bot::expert::choose_play_expert(p, me) {
            return cards;
        }
    }

    // Hard / Expert-fallback: time-boxed determinized search. The wall-clock budget
    // defaults to 1000ms but can be lowered via `SHENGJI_BOT_BUDGET_MS` (used by
    // the self-play eval harness to keep large runs fast). The cap guarantees a
    // slow CPU degrades to fewer simulations rather than hanging.
    if knobs.search_worlds > 0 {
        let config = SearchConfig {
            time_budget: Duration::from_millis(search_budget_ms()),
            max_worlds: knobs.search_worlds,
            max_candidates: knobs.search_candidates.max(1),
            rollout_tricks: knobs.rollout_tricks,
            seed: rng.gen(),
        };
        if let Some(cards) = search_play(p, me, config) {
            return cards;
        }
    }

    // Heuristic backbone (Easy, and the search tiers' fallback).
    let ranked: Vec<ScoredPlay> = if leading {
        heuristics::ranked_leads(p, me)
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
    _difficulty: BotDifficulty,
) -> Result<Option<Action>, Error> {
    if p.next_player()? != me || p.landlord() != me {
        // Only the landlord exchanges; nobody else acts during exchange.
        return Ok(None);
    }

    let trump = p.trump();

    // Determine how many cards we still need to bury. The kitty already holds
    // the correct number when the phase begins; if we want to bury *different*
    // cards we must first move some out and move our chosen ones in. Simpler and
    // robust: figure out our chosen burial set and reconcile one card at a time.
    let hand = p.hands().get(me).ok();
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
        if let Some(action) = reconcile_kitty(p, me, &hand_cards, trump) {
            return Ok(Some(action));
        }
    }

    // Friends (FindingFriends only; UI is Tractor-only but support it).
    let num_friends = p.num_friends();
    if num_friends > 0 {
        let friends = heuristics::choose_friends(trump, num_friends);
        if friends.len() == num_friends {
            return Ok(Some(Action::SetFriends(friends)));
        }
        // Fall back to legal side-suit aces if the heuristic came up short.
        let mut viable = vec![];
        for suit in &[Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            let c = Card::Suited {
                number: Number::Ace,
                suit: *suit,
            };
            if trump.effective_suit(c) != EffectiveSuit::Trump {
                viable.push(FriendSelection {
                    card: c,
                    initial_skip: 0,
                });
            }
        }
        if viable.len() >= num_friends {
            return Ok(Some(Action::SetFriends(viable[0..num_friends].to_vec())));
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
    // terminate.
    let mut pool: Vec<Card> = hand_cards.to_vec();
    pool.extend_from_slice(kitty);
    let desired = heuristics::choose_kitty(&pool, trump, kitty_size);

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
    if let Some(n) = card.number() {
        v += n.as_u32() as i32;
    }
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
) -> Option<shengji_mechanics::bidding::Bid> {
    let valid = p.valid_bids(me).ok()?;
    if valid.is_empty() {
        return None;
    }
    // The trump number for a bid is the bidder's own rank when there is no
    // landlord yet (which is the only situation where the driver invokes us).
    let level = p
        .propagated()
        .players()
        .iter()
        .find(|pl| pl.id == me)
        .map(|pl| pl.rank())
        .unwrap_or(Rank::Number(Number::Two));
    let hand: Vec<Card> = p
        .hands()
        .get(me)
        .ok()
        .map(|h| Card::cards(h.iter()).copied().collect())
        .unwrap_or_default();

    // Score each candidate bid by the trump it would establish.
    let mut best: Option<(f64, shengji_mechanics::bidding::Bid)> = None;
    for bid in valid {
        let candidate_trump = match bid.card {
            Card::SmallJoker | Card::BigJoker => heuristics::trump_for(level, None),
            Card::Suited { suit, .. } => heuristics::trump_for(level, Some(suit)),
            Card::Unknown => continue,
        };
        let mut strength = heuristics::bid_strength(&hand, candidate_trump);
        // Prefer fewer cards committed for the same strength (reinforce later).
        strength -= bid.count as f64 * 0.5;
        match &best {
            None => best = Some((strength, bid)),
            Some((bs, _)) if strength > *bs => best = Some((strength, bid)),
            _ => (),
        }
    }

    best.and_then(|(strength, bid)| {
        // Only bid if the hand is strong enough; otherwise pass.
        if strength >= 10.0 {
            Some(bid)
        } else {
            None
        }
    })
}

// ===========================================================================
// Always-legal fallback (the original dumb-but-legal policy)
// ===========================================================================

/// The original always-legal play used as a last-resort fallback so the bot
/// never produces an illegal / empty move when it must act.
fn dumb_play(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    if p.trick().played_cards().is_empty() {
        dumb_lead(p, me)
    } else {
        dumb_follow(p, me)
    }
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
        let results = TrickUnit::find_plays(trump, TractorRequirements::default(), suit_cards);
        let play = results
            .into_iter()
            .map(|play| play.into_iter().max_by_key(|u| u.size()).unwrap())
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
        .decomposition(Default::default())
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
