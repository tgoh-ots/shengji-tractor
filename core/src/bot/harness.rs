//! Shared headless self-play harness for the bot benchmarks in `core/examples/`.
//!
//! Every benchmark in `core/examples/` used to carry its OWN verbatim copy of the
//! seeded deal (`seeded_draw_phase`), the per-hand driver loop (`play_one_hand`),
//! and the honesty-respecting actor (`play_cards_for`). This module is the single
//! shared implementation, so a change to the deal / driver / honesty boundary
//! happens in ONE place and every benchmark stays consistent. The benchmarks now
//! differ only in how they configure the [`Seat`]s and how they aggregate the
//! [`HandResult`]s.
//!
//! # Honesty
//!
//! Exactly as in production ([`crate::bot::observed_state`]): only the
//! `Omniscient` CHEATER tier is ever handed the unredacted [`GameState`]; every
//! other brain acts from its own `GameState::for_player` redacted view.
//!
//! # Determinism
//!
//! The deal is fully determined by the `StdRng` passed to [`play_one_hand`] (the
//! ONLY consumer of that RNG), so two hands with the same seed get the SAME deal.
//! Search-LESS brains ([`PlayBrain::HeuristicDirect`], [`PlayBrain::EnochGreedy`],
//! [`PlayBrain::Easy`]) are then fully deterministic; search brains
//! ([`PlayBrain::Tier`] with a search tier, [`PlayBrain::Search`]) are
//! deterministic ONLY when their world cap binds before the time budget (so for a
//! byte-identical golden run, give them a very large `SHENGJI_BOT_BUDGET_MS`).
//!
//! # Paired evaluation
//!
//! [`run_paired_ab`] plays each deck seed in BOTH orientations (A-as-landlord and
//! B-as-landlord on the identical deal), which cancels the dominant deal-luck term
//! and resolves far smaller strength differences per game than the legacy
//! "alternate the landlord across DIFFERENT seeds" design. See
//! `docs/bot-training-roadmap.md` (the measurement substrate).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

use crate::bot::heuristics::{self, HeuristicVersion, ScoredPlay};
use crate::bot::search::{search_play, SearchConfig};
use crate::bot::{policy, BotDifficulty};
use crate::game_state::draw_phase::DrawPhase;
use crate::game_state::initialize_phase::InitializePhase;
use crate::game_state::play_phase::PlayPhase;
use crate::game_state::GameState;
use crate::interactive::Action;
use crate::settings::GameModeSettings;

/// How a seat plays the PLAY phase. Covers every variant the `core/examples`
/// benchmarks need (the production tiers, plus the search-less / explicit-config
/// probes used by the A/B harnesses).
#[derive(Clone, Copy, Debug)]
pub enum PlayBrain {
    /// A real difficulty tier driven through `policy::select_action` (the
    /// production path, including the determinized / perfect-info search). Honors
    /// the honesty boundary: only `Omniscient` sees the unredacted state.
    Tier(BotDifficulty),
    /// Greedy heuristic-direct play (NO search) at a specific scorer version.
    HeuristicDirect(HeuristicVersion),
    /// Greedy Enoch-playbook play (NO search).
    EnochGreedy,
    /// The determinized search called directly with an explicit [`SearchConfig`]
    /// (the per-decision `seed` is overwritten from the observable state). Honest
    /// (own redacted view); used by the budget A/B to vary the search config.
    Search(SearchConfig),
    /// The search-less Easy policy with explicit knobs (ε blunder rate + softmax
    /// temperature), used by the Easy knob A/B. Honest (own redacted view).
    Easy { epsilon: f64, temperature: f64 },
}

/// One seat's full configuration: how it plays, bids, and buries the kitty. The
/// `bid` / `kitty` difficulties are kept separate because some benchmarks bid and
/// bury with a DIFFERENT tier than they play with (e.g. the heuristic A/B bids
/// Expert but buries Easy so only the play scorer differs).
#[derive(Clone, Copy, Debug)]
pub struct Seat {
    pub play: PlayBrain,
    /// Difficulty used for `policy::choose_bid` on this seat.
    pub bid: BotDifficulty,
    /// Difficulty used for the exchange/kitty decisions when this seat is the
    /// landlord.
    pub kitty: BotDifficulty,
}

impl Seat {
    /// A seat that plays, bids, and buries as a single difficulty tier (the
    /// common case for the tier tournament / ladder benchmarks).
    pub fn tier(d: BotDifficulty) -> Self {
        Seat {
            play: PlayBrain::Tier(d),
            bid: d,
            kitty: d,
        }
    }
}

/// A named contestant = one partnership's [`Seat`] config (both of its two seats
/// play identically) plus a display label for reports.
#[derive(Clone, Debug)]
pub struct Contestant {
    pub label: String,
    pub seat: Seat,
}

impl Contestant {
    pub fn new(label: impl Into<String>, seat: Seat) -> Self {
        Contestant {
            label: label.into(),
            seat,
        }
    }

    /// A contestant that plays/bids/buries as a single tier.
    pub fn tier(d: BotDifficulty) -> Self {
        Contestant::new(d.as_str(), Seat::tier(d))
    }
}

/// The outcome of one finished hand, in seat-relative terms. Seat 0 is always the
/// landlord, so the landlord TEAM is seats 0 & 2 and the attacking team is 1 & 3.
#[derive(Clone, Copy, Debug)]
pub struct HandResult {
    pub landlord_won: bool,
    pub landlord_seat: PlayerID,
    /// The attacking (non-landlord) team's captured points — the margin signal.
    pub non_landlord_points: isize,
}

impl HandResult {
    /// `(won, margin)` from the perspective of the partnership that occupies the
    /// landlord team iff `subject_is_landlord_team`. Attackers want
    /// `non_landlord_points` HIGH, defenders want it LOW, so the margin is
    /// oriented so higher is always better for the subject.
    pub fn subject_outcome(&self, subject_is_landlord_team: bool) -> (bool, isize) {
        let won = self.landlord_won == subject_is_landlord_team;
        let margin = if subject_is_landlord_team {
            -self.non_landlord_points
        } else {
            self.non_landlord_points
        };
        (won, margin)
    }
}

/// Build a fully-seeded 4-player, 2-deck Tractor Draw phase with seat 0
/// preselected as landlord. We construct the `DrawPhase` directly from a
/// seed-shuffled deck so the ENTIRE game is reproducible (the engine's own
/// `InitializePhase::start` uses `thread_rng`, which we cannot seed). This is the
/// shared deal every seeded benchmark uses.
pub fn seeded_draw_phase(decks: &[Deck], rng: &mut StdRng) -> DrawPhase {
    let mut deck: Vec<_> = decks.iter().flat_map(|d| d.cards()).collect();
    deck.shuffle(rng);

    let num_players = 4;
    let mut kitty_size = deck.len() % num_players;
    if kitty_size == 0 {
        kitty_size = num_players;
    }
    if kitty_size < 5 {
        kitty_size += num_players;
    }

    let mut init = InitializePhase::new();
    for i in 0..num_players {
        init.add_player(format!("seat{i}")).unwrap();
    }
    init.set_num_decks(Some(decks.len())).unwrap();
    init.set_game_mode(GameModeSettings::Tractor).unwrap();
    let real_seats: Vec<PlayerID> = init.players().iter().map(|p| p.id).collect();
    init.set_landlord(Some(real_seats[0])).unwrap();
    let propagated = (*init).clone();

    let level = Some(propagated.players()[0].rank());
    let hands_deck = deck[0..deck.len() - kitty_size].to_vec();
    let kitty = deck[deck.len() - kitty_size..].to_vec();

    DrawPhase::new(
        propagated,
        0,
        hands_deck,
        kitty,
        decks.len(),
        crate::settings::GameMode::Tractor,
        level,
        decks.to_vec(),
        vec![],
    )
}

/// A per-decision RNG seed derived ONLY from the observable position (acting seat,
/// its hand size, cards already on the table). Matches `policy::rng_for` and the
/// per-decision seeds the budget / Easy A/B harnesses used, so a search/Easy brain
/// is reproducible given the deal.
fn decision_seed(p: &PlayPhase, me: PlayerID) -> u64 {
    let hand_size = p
        .hands()
        .get(me)
        .map(|h| h.values().sum::<usize>())
        .unwrap_or(0);
    let on_table = p.trick().played_cards().len();
    (me.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (hand_size as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (on_table as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
}

/// Reproduce the search-less Easy policy for `actor` under `(epsilon, temperature)`
/// from its own redacted view (mirrors `policy::choose_play` for the
/// `search_worlds == 0` tier and the legacy `easy_ab_benchmark`). Returns `None`
/// only if no legal candidate exists.
fn easy_play(
    p: &PlayPhase,
    actor: PlayerID,
    epsilon: f64,
    temperature: f64,
    rng: &mut StdRng,
) -> Option<Vec<Card>> {
    let leading = p.trick().played_cards().is_empty();
    let ranked: Vec<ScoredPlay> = if leading {
        heuristics::ranked_leads(p, actor)
    } else {
        heuristics::ranked_follows(p, actor)
    };
    if ranked.is_empty() {
        return None;
    }
    // ε-blunder: a uniformly random legal candidate.
    if rng.gen_bool(epsilon.clamp(0.0, 1.0)) {
        let idx = rng.gen_range(0..ranked.len());
        return Some(ranked[idx].cards.clone());
    }
    // Otherwise softmax-sample the top-4 candidates at temperature T.
    if temperature <= 0.0 {
        return Some(ranked[0].cards.clone());
    }
    let top = &ranked[..ranked.len().min(4)];
    let max = top.iter().map(|c| c.score).fold(f64::MIN, f64::max);
    let weights: Vec<f64> = top
        .iter()
        .map(|c| ((c.score - max) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Some(top[0].cards.clone());
    }
    let mut pick = rng.gen::<f64>() * total;
    for (c, w) in top.iter().zip(weights.iter()) {
        pick -= w;
        if pick <= 0.0 {
            return Some(c.cards.clone());
        }
    }
    Some(top[top.len() - 1].cards.clone())
}

/// Choose the PLAY-phase cards for `actor` under its `brain`, honoring the honesty
/// boundary (only `Omniscient` sees the unredacted state).
pub fn play_cards_for(s: &PlayPhase, actor: PlayerID, brain: &PlayBrain) -> Option<Vec<Card>> {
    match brain {
        PlayBrain::Tier(d) => {
            let view = if matches!(d, BotDifficulty::Omniscient) {
                GameState::Play(s.clone())
            } else {
                GameState::Play(s.clone()).for_player(actor)
            };
            match policy::select_action(&view, actor, *d).ok()? {
                Some(Action::PlayCards(c)) => Some(c),
                _ => None,
            }
        }
        PlayBrain::HeuristicDirect(version) => {
            let view = GameState::Play(s.clone()).for_player(actor);
            match &view {
                GameState::Play(pp) => heuristics::choose_play_direct(pp, actor, *version),
                _ => None,
            }
        }
        PlayBrain::EnochGreedy => {
            let view = GameState::Play(s.clone()).for_player(actor);
            match &view {
                GameState::Play(pp) => heuristics::choose_play_direct_enoch(pp, actor),
                _ => None,
            }
        }
        PlayBrain::Search(cfg) => {
            let view = GameState::Play(s.clone()).for_player(actor);
            let pp = match &view {
                GameState::Play(pp) => pp,
                _ => return None,
            };
            let mut config = *cfg;
            config.seed = decision_seed(pp, actor);
            if let Some(c) = search_play(pp, actor, config) {
                return Some(c);
            }
            // Same config-independent fallback the live Expert policy uses, so a
            // degenerate position never stalls the harness (and can't bias an A/B).
            match policy::select_action(&view, actor, BotDifficulty::Expert).ok()? {
                Some(Action::PlayCards(c)) => Some(c),
                _ => None,
            }
        }
        PlayBrain::Easy {
            epsilon,
            temperature,
        } => {
            let view = GameState::Play(s.clone()).for_player(actor);
            let pp = match &view {
                GameState::Play(pp) => pp,
                _ => return None,
            };
            let mut rng = StdRng::seed_from_u64(decision_seed(pp, actor));
            easy_play(pp, actor, *epsilon, *temperature, &mut rng)
        }
    }
}

/// Drive a single seeded 4-player Tractor hand to completion. `seats[i]` controls
/// seat `i` (seat 0 is the landlord). The deal is determined entirely by `rng`.
/// Returns `None` only on an unexpected engine error (which would itself be a bug)
/// or if the iteration cap is hit.
pub fn play_one_hand(seats: &[Seat; 4], rng: &mut StdRng) -> Option<HandResult> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seat_ids: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();
    let seat_idx = |pid: PlayerID| -> Option<usize> { seat_ids.iter().position(|x| *x == pid) };

    let mut state = GameState::Draw(draw);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = s.next_player().ok()?;
                    s.draw_card(p).ok()?;
                } else if s.bid_decided() {
                    let responsible = s.next_player().ok()?;
                    state = GameState::Exchange(s.advance(responsible).ok()?);
                } else {
                    // Each seat bids by its own configured bid difficulty; the
                    // first seat that wants and can place a bid takes it.
                    let mut bid = false;
                    for (idx, &seat) in seat_ids.iter().enumerate() {
                        if let Some(b) = policy::choose_bid(s, seat, seats[idx].bid) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seat_ids {
                            if let Some(b) =
                                s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count)
                            {
                                if s.bid(seat, b.card, b.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(s) => {
                let landlord = s.landlord();
                let d = seats[seat_idx(landlord)?].kitty;
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                match policy::select_action(&view, landlord, d).ok()? {
                    Some(Action::MoveCardToKitty(c)) => s.move_card_to_kitty(landlord, c).ok()?,
                    Some(Action::MoveCardToHand(c)) => s.move_card_to_hand(landlord, c).ok()?,
                    Some(Action::SetFriends(f)) => s.set_friends(landlord, f).ok()?,
                    _ => state = GameState::Play(s.advance(landlord).ok()?),
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let landlord_seat = s.landlord();
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;
                    return Some(HandResult {
                        landlord_won,
                        landlord_seat,
                        non_landlord_points,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let cards = play_cards_for(s, actor, &seats[seat_idx(actor)?].play)?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Paired-on-mirrored-deck A/B + statistics (the measurement substrate).
// ===========================================================================

/// The result of a paired A-vs-B match. Each deck seed is played in BOTH
/// orientations (A as the landlord team, then B as the landlord team, on the
/// IDENTICAL deal), so deal luck is paired out. Per-deck observations (A's mean
/// win indicator and mean margin across the two orientations) are the independent
/// units for the bootstrap, which is why this resolves much smaller effects than
/// the legacy unpaired design.
#[derive(Clone, Debug)]
pub struct PairedABResult {
    pub label_a: String,
    pub label_b: String,
    /// Number of deck seeds attempted (each played in both orientations).
    pub pairs: usize,
    /// Number of fully-completed individual hands (≤ `2 * pairs`).
    pub completed_hands: usize,
    /// Number of decks where BOTH orientations completed (the paired units).
    pub complete_pairs: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    /// Per-deck mean win indicator for A across its two orientations (∈ {0, .5, 1}).
    pub per_deck_winrate: Vec<f64>,
    /// Per-deck mean point margin for A across its two orientations.
    pub per_deck_margin: Vec<f64>,
}

impl PairedABResult {
    /// A's overall win-rate over all completed individual hands.
    pub fn win_rate(&self) -> f64 {
        let n = self.a_wins + self.b_wins;
        if n == 0 {
            0.0
        } else {
            self.a_wins as f64 / n as f64
        }
    }

    /// A's mean per-deck win-rate (the paired point estimate).
    pub fn paired_win_rate(&self) -> f64 {
        mean(&self.per_deck_winrate)
    }

    /// A's mean per-deck point margin.
    pub fn paired_margin(&self) -> f64 {
        mean(&self.per_deck_margin)
    }

    /// Percentile bootstrap 95% CI for A's paired win-rate, resampling over DECKS
    /// (the paired units), so the interval respects the pairing.
    pub fn winrate_bootstrap_ci(&self) -> (f64, f64) {
        bootstrap_mean_ci(&self.per_deck_winrate, 2000, 0xB007_5EED)
    }

    /// Wilson score 95% interval for the win-rate over the individual completed
    /// hands (a simpler, pairing-agnostic interval; report alongside the bootstrap).
    pub fn winrate_wilson_ci(&self) -> (f64, f64) {
        wilson_interval(self.a_wins, self.a_wins + self.b_wins, 1.96)
    }

    /// The 95% minimum detectable effect on A's paired win-rate: the half-width of
    /// the normal CI of the per-deck mean. An observed |win-rate − 0.5| below this
    /// is NOT resolvable at this sample size ("underpowered", not "no difference").
    pub fn winrate_mde(&self) -> f64 {
        ci_half_width(&self.per_deck_winrate, 1.96)
    }
}

/// Run a paired A-vs-B match over `pairs` deck seeds (each played in both
/// orientations on the identical deal). `base_seed` selects the deck-seed
/// sequence.
pub fn run_paired_ab(
    a: &Contestant,
    b: &Contestant,
    pairs: usize,
    base_seed: u64,
) -> PairedABResult {
    let mut a_wins = 0usize;
    let mut b_wins = 0usize;
    let mut completed_hands = 0usize;
    let mut complete_pairs = 0usize;
    let mut per_deck_winrate = Vec::with_capacity(pairs);
    let mut per_deck_margin = Vec::with_capacity(pairs);

    for d in 0..pairs {
        let seed = base_seed.wrapping_add(d as u64);
        // Orientation 1: A is the landlord team (seats 0,2); B attacks (1,3).
        let mut rng1 = StdRng::seed_from_u64(seed);
        let seats1 = [a.seat, b.seat, a.seat, b.seat];
        let r1 = play_one_hand(&seats1, &mut rng1);
        // Orientation 2: SAME deal (same seed), roles swapped — B is landlord.
        let mut rng2 = StdRng::seed_from_u64(seed);
        let seats2 = [b.seat, a.seat, b.seat, a.seat];
        let r2 = play_one_hand(&seats2, &mut rng2);

        let mut wins = Vec::with_capacity(2);
        let mut margins = Vec::with_capacity(2);
        if let Some(r1) = r1 {
            // A is the landlord team in orientation 1.
            let (won, margin) = r1.subject_outcome(true);
            if won {
                a_wins += 1;
            } else {
                b_wins += 1;
            }
            completed_hands += 1;
            wins.push(if won { 1.0 } else { 0.0 });
            margins.push(margin as f64);
        }
        if let Some(r2) = r2 {
            // A is the ATTACKER team in orientation 2.
            let (won, margin) = r2.subject_outcome(false);
            if won {
                a_wins += 1;
            } else {
                b_wins += 1;
            }
            completed_hands += 1;
            wins.push(if won { 1.0 } else { 0.0 });
            margins.push(margin as f64);
        }
        // Only fully-paired decks (both orientations completed) become bootstrap
        // units, so the pairing is exact.
        if wins.len() == 2 {
            complete_pairs += 1;
            per_deck_winrate.push((wins[0] + wins[1]) / 2.0);
            per_deck_margin.push((margins[0] + margins[1]) / 2.0);
        }
    }

    PairedABResult {
        label_a: a.label.clone(),
        label_b: b.label.clone(),
        pairs,
        completed_hands,
        complete_pairs,
        a_wins,
        b_wins,
        per_deck_winrate,
        per_deck_margin,
    }
}

/// Pretty-print a [`PairedABResult`] (win-rate, both CIs, margin, MDE).
pub fn print_paired_ab(r: &PairedABResult) {
    let (wlo, whi) = r.winrate_wilson_ci();
    let (blo, bhi) = r.winrate_bootstrap_ci();
    println!(
        "=== {} vs {}  ({} decks × 2 orientations = {} hands, {} paired) ===",
        r.label_a, r.label_b, r.pairs, r.completed_hands, r.complete_pairs,
    );
    println!(
        "  {} win-rate: {:.1}%  (Wilson95 [{:.1}, {:.1}]; paired-bootstrap95 [{:.1}, {:.1}])",
        r.label_a,
        r.win_rate() * 100.0,
        wlo * 100.0,
        whi * 100.0,
        blo * 100.0,
        bhi * 100.0,
    );
    println!(
        "  {} paired margin: {:+.2} pts/hand   95% MDE on win-rate: ±{:.1}pp",
        r.label_a,
        r.paired_margin(),
        r.winrate_mde() * 100.0,
    );
}

/// Mean of a slice (0.0 for empty).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Sample standard deviation (0.0 for fewer than 2 elements).
pub fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0);
    var.sqrt()
}

/// Half-width of the normal `z`-CI of the MEAN of `xs` (= z · SE). Used as the
/// minimum detectable effect.
pub fn ci_half_width(xs: &[f64], z: f64) -> f64 {
    if xs.len() < 2 {
        return f64::INFINITY;
    }
    z * std_dev(xs) / (xs.len() as f64).sqrt()
}

/// Wilson score interval for a binomial proportion `wins / n` at the given `z`
/// (use 1.96 for 95%). Returns `(lo, hi)` clamped to `[0, 1]`; `(0, 0)` for `n=0`.
/// More accurate than the normal approximation for small `n` / extreme rates.
pub fn wilson_interval(wins: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let n = n as f64;
    let phat = wins as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (phat + z2 / (2.0 * n)) / denom;
    let half = (z * ((phat * (1.0 - phat) / n + z2 / (4.0 * n * n)).sqrt())) / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// Percentile bootstrap 95% CI for the MEAN of `xs`, resampling WITH replacement
/// `iters` times using a fixed `seed` (reproducible). Returns `(p2.5, p97.5)`;
/// `(mean, mean)` for fewer than 2 elements.
pub fn bootstrap_mean_ci(xs: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    if xs.len() < 2 {
        let m = mean(xs);
        return (m, m);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = xs.len();
    let mut means: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut acc = 0.0;
        for _ in 0..n {
            acc += xs[rng.gen_range(0..n)];
        }
        means.push(acc / n as f64);
    }
    means.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[((iters as f64) * 0.025) as usize];
    let hi = means[(((iters as f64) * 0.975) as usize).min(iters - 1)];
    (lo, hi)
}
