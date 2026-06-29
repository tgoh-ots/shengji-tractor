//! Full round-robin tournament across the bot TIERS.
//!
//! Plays every unordered pairing of [Easy, Expert, Enoch, Omniscient] (6 pairings)
//! as a seeded head-to-head match. In each match one partnership (2 seats) plays as
//! tier A and the other as tier B; we ALTERNATE which partnership is the
//! landlord/defending team across games to cancel the dealer/positional bias, and
//! we reuse the same seeds across reversed pairings so the matrix is symmetric on
//! the deals. We report a clean WIN-RATE MATRIX (each tier's win-rate vs each other
//! tier) plus average point margins, and the implied ladder ordering.
//!
//! This reuses `enoch_benchmark`'s seeded game-driving harness verbatim (the
//! `seeded_draw_phase` deal loop, the `Brain::Tier(BotDifficulty)` play/declare
//! path, and the per-hand driver). Every tier here is a real difficulty tier driven
//! through `policy::select_action`; the honesty boundary is preserved (only
//! Omniscient sees the unredacted state).
//!
//! Search budget is set via `SHENGJI_BOT_BUDGET_MS` (the example defaults it to
//! 100ms if unset, for speed).
//!
//! Run with:
//!   cargo run --release --example tournament -- [games_per_pairing] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_core::settings::GameModeSettings;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

/// The four tiers, in nominal ladder order (weakest -> strongest).
const TIERS: [BotDifficulty; 4] = [
    BotDifficulty::Easy,
    BotDifficulty::Expert,
    BotDifficulty::Enoch,
    BotDifficulty::Omniscient,
];

fn tier_label(d: BotDifficulty) -> &'static str {
    d.as_str()
}

/// The outcome of one finished hand, from tier A's (the "subject") perspective.
struct GameOutcome {
    a_won: bool,
    a_point_margin: isize,
}

/// Build a fully-seeded 4-player, 2-deck Tractor Draw phase, seat 0 preselected as
/// landlord (verbatim from `enoch_benchmark::seeded_draw_phase`).
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

/// Pick the play-phase cards for `actor` playing tier `d`, honoring the honesty
/// boundary (Omniscient sees the true state; everyone else only their own redacted
/// view). Mirrors `enoch_benchmark::play_cards_for`'s `Brain::Tier` arm.
fn play_cards_for(
    s: &shengji_core::game_state::play_phase::PlayPhase,
    actor: PlayerID,
    d: BotDifficulty,
) -> Option<Vec<Card>> {
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

/// Drive one seeded hand: tier `a` vs tier `b`. `a_is_landlord_team` selects which
/// partnership plays tier `a`; the other plays tier `b`. The result is always from
/// tier `a`'s perspective. (Verbatim driver from `enoch_benchmark::play_one_hand`,
/// generalized to two arbitrary tiers.)
fn play_one_hand(
    a_is_landlord_team: bool,
    a: BotDifficulty,
    b: BotDifficulty,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    // Seats 0,2 are the landlord (defending) team; 1,3 attack. Tier `a` occupies
    // the landlord team iff `a_is_landlord_team`.
    let tier_of = |seat_idx: usize| -> BotDifficulty {
        let is_landlord_team = seat_idx % 2 == 0;
        if is_landlord_team == a_is_landlord_team {
            a
        } else {
            b
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
                    // Each seat bids by its tier's (difficulty-aware) bid policy.
                    let mut bid = false;
                    for (idx, &seat) in seats.iter().enumerate() {
                        let d = tier_of(idx);
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
                let d = tier_of(landlord_idx);
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

                    // Tier `a` is the defending (landlord) team iff a_is_landlord_team.
                    let a_is_defender = a_is_landlord_team;
                    let a_won = landlord_won == a_is_defender;
                    let a_point_margin = if a_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };
                    return Some(GameOutcome {
                        a_won,
                        a_point_margin,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let cards = play_cards_for(s, actor, tier_of(actor_idx))?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

/// Aggregate result of one head-to-head pairing, from tier A's perspective.
struct PairResult {
    a: BotDifficulty,
    b: BotDifficulty,
    completed: usize,
    a_wins: usize,
    a_total_margin: isize,
}

/// Run a full `a`-vs-`b` match over `num_games` seeded hands, alternating which
/// partnership plays tier `a` to cancel positional bias.
fn run_pair(a: BotDifficulty, b: BotDifficulty, num_games: usize, base_seed: u64) -> PairResult {
    let mut a_wins = 0usize;
    let mut a_total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let a_is_landlord_team = g % 2 == 0;
        if let Some(outcome) = play_one_hand(a_is_landlord_team, a, b, &mut rng) {
            completed += 1;
            if outcome.a_won {
                a_wins += 1;
            }
            a_total_margin += outcome.a_point_margin;
        }
    }

    PairResult {
        a,
        b,
        completed,
        a_wins,
        a_total_margin,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let games_per_pairing: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(160);
    let base_seed: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x70_75_72);

    // Default the search budget to 100ms for speed if the caller didn't set it.
    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "100");
    }
    let budget = env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default();

    println!("ROUND-ROBIN TIER TOURNAMENT");
    println!(
        "Tiers: {}",
        TIERS
            .iter()
            .map(|d| tier_label(*d))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Games per pairing: {games_per_pairing}  base_seed: {base_seed:#x}  budget_ms: {budget}"
    );
    println!(
        "Each pairing alternates which partnership is the landlord/defending team \
         across games to cancel positional bias. Honesty preserved (only Omniscient \
         sees the unredacted state).\n"
    );

    let n = TIERS.len();
    // win_rate[i][j] = tier i's win-rate (%) vs tier j; margin[i][j] = avg pts/game.
    let mut win_rate = vec![vec![f64::NAN; n]; n];
    let mut margin = vec![vec![f64::NAN; n]; n];
    let mut wins_for = vec![0usize; n]; // total wins across all this tier's games
    let mut games_for = vec![0usize; n]; // total games this tier played

    let overall_start = Instant::now();

    for i in 0..n {
        for j in (i + 1)..n {
            let a = TIERS[i];
            let b = TIERS[j];
            let start = Instant::now();
            let r = run_pair(a, b, games_per_pairing, base_seed);
            let a_wr = if r.completed > 0 {
                r.a_wins as f64 / r.completed as f64 * 100.0
            } else {
                0.0
            };
            let a_marg = r.a_total_margin as f64 / r.completed.max(1) as f64;
            let b_wins = r.completed - r.a_wins;
            let b_wr = 100.0 - a_wr; // every completed game has exactly one winner
            let b_marg = -a_marg; // margin is zero-sum between the two partnerships

            win_rate[i][j] = a_wr;
            win_rate[j][i] = b_wr;
            margin[i][j] = a_marg;
            margin[j][i] = b_marg;

            wins_for[i] += r.a_wins;
            games_for[i] += r.completed;
            wins_for[j] += b_wins;
            games_for[j] += r.completed;

            println!(
                "=== {} vs {} ({} games) ===",
                tier_label(r.a),
                tier_label(r.b),
                r.completed
            );
            println!(
                "  {} win-rate: {:.2}%  ({} wins)   avg margin {:+.2} pts/game",
                tier_label(r.a),
                a_wr,
                r.a_wins,
                a_marg
            );
            println!(
                "  {} win-rate: {:.2}%  ({} wins)   avg margin {:+.2} pts/game",
                tier_label(r.b),
                b_wr,
                b_wins,
                b_marg
            );
            println!("  Elapsed: {:.1}s\n", start.elapsed().as_secs_f64());
        }
    }

    // ---- WIN-RATE MATRIX ----
    println!("================ WIN-RATE MATRIX (row tier's win-% vs column tier) ================");
    print!("{:>12}", "");
    for d in TIERS.iter() {
        print!("{:>12}", tier_label(*d));
    }
    println!("{:>10}", "OVERALL");
    for i in 0..n {
        print!("{:>12}", tier_label(TIERS[i]));
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>11.1}%", win_rate[i][j]);
            }
        }
        let overall = if games_for[i] > 0 {
            wins_for[i] as f64 / games_for[i] as f64 * 100.0
        } else {
            0.0
        };
        println!("{:>9.1}%", overall);
    }

    // ---- POINT-MARGIN MATRIX ----
    println!(
        "\n========= AVG POINT-MARGIN MATRIX (row tier's avg pts/game vs column tier) ========="
    );
    print!("{:>12}", "");
    for d in TIERS.iter() {
        print!("{:>12}", tier_label(*d));
    }
    println!();
    for i in 0..n {
        print!("{:>12}", tier_label(TIERS[i]));
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>+12.2}", margin[i][j]);
            }
        }
        println!();
    }

    // ---- IMPLIED LADDER (by overall win-rate across all games played) ----
    let mut ranking: Vec<(BotDifficulty, f64)> = (0..n)
        .map(|i| {
            let wr = if games_for[i] > 0 {
                wins_for[i] as f64 / games_for[i] as f64 * 100.0
            } else {
                0.0
            };
            (TIERS[i], wr)
        })
        .collect();
    ranking.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    println!("\n================ IMPLIED LADDER (by overall win-rate) ================");
    for (rank, (d, wr)) in ranking.iter().enumerate() {
        println!("  {}. {:<11} {:.1}% overall", rank + 1, tier_label(*d), wr);
    }
    let order = ranking
        .iter()
        .map(|(d, _)| tier_label(*d))
        .collect::<Vec<_>>()
        .join(" > ");
    println!("\n  Implied ordering (strongest -> weakest): {order}");
    println!(
        "\nTotal tournament elapsed: {:.1}s",
        overall_start.elapsed().as_secs_f64()
    );
}
