//! Headless benchmark for the ENOCH bot tier.
//!
//! One partnership (2 seats) plays the whole hand as ENOCH — its pair-prioritized
//! trump declaration, point-budgeted kitty burial, and the Enoch-playbook play
//! phase (tractor-first / long-suit leading, defender low-trump hand-off, endgame
//! kitty protection). The other partnership plays as a chosen OPPONENT tier. We
//! alternate which partnership is Enoch across games to cancel the dealer/landlord
//! positional edge, and report Enoch's win-rate (and average point margin) vs each
//! opponent.
//!
//! The opponents:
//!   * NewDefault — the new default boss-/partner-aware heuristic, GREEDY (no
//!     search). This is the play scorer every honest tier shares; beating it shows
//!     the Enoch playbook adds value on top of the improved heuristic.
//!   * Hard       — the heuristic PLUS the time-boxed determinized search.
//!   * Omniscient — the perfect-information CHEATER (an upper bound; Enoch is
//!     honest, so it is expected to LOSE to Omniscient — we report how close).
//!
//! For speed the Enoch / NewDefault play is GREEDY (`choose_play_direct*`, no
//! search); Hard / Omniscient run their real (search-based) policy with a small
//! per-decision budget. The declare / kitty / endgame RULES apply deterministically
//! regardless. Set `SHENGJI_BOT_BUDGET_MS` to trade strength for speed in the
//! search tiers.
//!
//! Run with:
//!   cargo run --release --example enoch_benchmark -- [num_games] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_core::settings::GameModeSettings;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

/// How a partnership plays a hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Brain {
    /// Full Enoch playbook (pair-aware declare + point-budgeted kitty + Enoch
    /// play). GREEDY play (no search) — fast, apples-to-apples vs `NewDefault`.
    EnochGreedy,
    /// The new default boss-/partner-aware heuristic, GREEDY play (no search).
    NewDefault,
    /// A real difficulty tier driven through `policy::select_action` (search).
    /// `Tier(Enoch)` is the REAL Enoch tier (playbook + determinized search) —
    /// the fair opponent for the other search tiers.
    Tier(BotDifficulty),
}

impl Brain {
    fn label(self) -> String {
        match self {
            Brain::EnochGreedy => "Enoch(greedy)".to_string(),
            Brain::NewDefault => "NewDefault(greedy)".to_string(),
            Brain::Tier(d) => format!("{d:?}(search)"),
        }
    }

    /// The difficulty to use for the shared declare/kitty driver decisions
    /// (bidding + burial), which are difficulty-aware.
    fn declare_difficulty(self) -> BotDifficulty {
        match self {
            Brain::EnochGreedy => BotDifficulty::Enoch,
            Brain::NewDefault => BotDifficulty::Hard,
            Brain::Tier(d) => d,
        }
    }
}

/// The outcome of one finished hand, from Enoch's perspective.
struct GameOutcome {
    enoch_won: bool,
    enoch_point_margin: isize,
}

/// Build a fully-seeded 4-player, 2-deck Tractor Draw phase, seat 0 preselected as
/// landlord (mirrors `heuristic_benchmark`'s `seeded_draw_phase`).
fn seeded_draw_phase(decks: &[Deck], rng: &mut StdRng) -> DrawPhase {
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
        shengji_core::settings::GameMode::Tractor,
        level,
        decks.to_vec(),
        vec![],
    )
}

/// Pick the play-phase cards for `actor` under its partnership `brain`, honoring
/// the honesty boundary (Omniscient sees the true state; everyone else only their
/// own redacted view).
fn play_cards_for(
    s: &shengji_core::game_state::play_phase::PlayPhase,
    actor: PlayerID,
    brain: Brain,
) -> Option<Vec<Card>> {
    match brain {
        Brain::EnochGreedy => {
            let view = GameState::Play(s.clone()).for_player(actor);
            match &view {
                GameState::Play(pp) => heuristics::choose_play_direct_enoch(pp, actor),
                _ => None,
            }
        }
        Brain::NewDefault => {
            let view = GameState::Play(s.clone()).for_player(actor);
            match &view {
                GameState::Play(pp) => {
                    heuristics::choose_play_direct(pp, actor, HeuristicVersion::New)
                }
                _ => None,
            }
        }
        Brain::Tier(d) => {
            // Honesty bypass: only Omniscient sees the unredacted state.
            let view = if matches!(d, BotDifficulty::Omniscient) {
                GameState::Play(s.clone())
            } else {
                GameState::Play(s.clone()).for_player(actor)
            };
            match policy::select_action(&view, actor, d).ok()? {
                Some(Action::PlayCards(c)) => Some(c),
                _ => None,
            }
        }
    }
}

/// Drive one seeded hand. `enoch_is_landlord_team` selects which partnership is
/// the `enoch` brain; the other plays `opponent`. The `enoch_point_margin` /
/// `enoch_won` in the result are always from the `enoch`-brain partnership's
/// perspective.
fn play_one_hand(
    enoch_is_landlord_team: bool,
    enoch: Brain,
    opponent: Brain,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    // Seats 0,2 are the landlord (defending) team; 1,3 attack. The `enoch` brain
    // occupies the landlord team iff `enoch_is_landlord_team`.
    let brain_of = |seat_idx: usize| -> Brain {
        let is_landlord_team = seat_idx % 2 == 0;
        if is_landlord_team == enoch_is_landlord_team {
            enoch
        } else {
            opponent
        }
    };

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
                    // Each seat bids by its partnership's declare difficulty (so
                    // Enoch's pair-aware, not-too-early bidding actually drives the
                    // trump it picks).
                    let mut bid = false;
                    for (idx, &seat) in seats.iter().enumerate() {
                        let d = brain_of(idx).declare_difficulty();
                        if let Some(b) = policy::choose_bid(s, seat, d) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seats {
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
                let landlord_idx = seats.iter().position(|x| *x == landlord)?;
                let d = brain_of(landlord_idx).declare_difficulty();
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
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;

                    // Enoch is the defending (landlord) team iff enoch_is_landlord_team.
                    let enoch_is_defender = enoch_is_landlord_team;
                    let enoch_won = landlord_won == enoch_is_defender;
                    let enoch_point_margin = if enoch_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };
                    return Some(GameOutcome {
                        enoch_won,
                        enoch_point_margin,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let cards = play_cards_for(s, actor, brain_of(actor_idx))?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

/// Run a full `enoch`-vs-`opponent` match and print the result (from the
/// `enoch` brain's perspective).
fn run_match(enoch: Brain, opponent: Brain, num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let mut enoch_wins = 0usize;
    let mut opp_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;
    let mut enoch_as_landlord_games = 0usize;
    let mut enoch_as_landlord_wins = 0usize;
    let mut enoch_as_attacker_games = 0usize;
    let mut enoch_as_attacker_wins = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let enoch_is_landlord_team = g % 2 == 0;
        match play_one_hand(enoch_is_landlord_team, enoch, opponent, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.enoch_won {
                    enoch_wins += 1;
                } else {
                    opp_wins += 1;
                }
                total_margin += outcome.enoch_point_margin;
                if enoch_is_landlord_team {
                    enoch_as_landlord_games += 1;
                    if outcome.enoch_won {
                        enoch_as_landlord_wins += 1;
                    }
                } else {
                    enoch_as_attacker_games += 1;
                    if outcome.enoch_won {
                        enoch_as_attacker_wins += 1;
                    }
                }
            }
            None => eprintln!("  game {g}: engine error / skipped"),
        }
    }

    let win_rate = if completed > 0 {
        enoch_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "=== {} vs {} ({completed} games) ===",
        enoch.label(),
        opponent.label()
    );
    println!("  Enoch wins:    {enoch_wins}");
    println!("  Opponent wins: {opp_wins}");
    println!("  Enoch win-rate:           {win_rate:.2}%");
    println!(
        "  Enoch avg point margin:   {:+.2} pts/game",
        total_margin as f64 / completed.max(1) as f64
    );
    if enoch_as_landlord_games > 0 {
        println!(
            "    as LANDLORD: {}/{} = {:.1}%",
            enoch_as_landlord_wins,
            enoch_as_landlord_games,
            enoch_as_landlord_wins as f64 / enoch_as_landlord_games as f64 * 100.0
        );
    }
    if enoch_as_attacker_games > 0 {
        println!(
            "    as ATTACKER: {}/{} = {:.1}%",
            enoch_as_attacker_wins,
            enoch_as_attacker_games,
            enoch_as_attacker_wins as f64 / enoch_as_attacker_games as f64 * 100.0
        );
    }
    println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xE0C);

    println!("ENOCH tier benchmark");
    println!("Games per match: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Match 1 is greedy-vs-greedy (apples-to-apples on the play scorer); matches \
         2-3 run the REAL Enoch tier (playbook + determinized search) against the \
         other search tiers. Declare/kitty/endgame rules apply deterministically.\n"
    );

    // 1) Enoch-greedy vs the new-default greedy heuristic (the shared play scorer)
    //    — isolates the value of the Enoch playbook on top of the heuristic.
    run_match(Brain::EnochGreedy, Brain::NewDefault, num_games, base_seed);
    // 2) The REAL Enoch tier (search) vs Hard (search) — search-vs-search.
    run_match(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Hard),
        num_games,
        base_seed,
    );
    // 3) The REAL Enoch tier (search, HONEST) vs the perfect-information
    //    Omniscient cheater — an upper bound; Enoch is expected to lose, we report
    //    how close it stays.
    run_match(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Omniscient),
        num_games,
        base_seed,
    );
}
