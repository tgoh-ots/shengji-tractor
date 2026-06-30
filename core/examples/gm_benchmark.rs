//! Head-to-head benchmark for the GRANDMASTER honest tier.
//!
//! Pits two real difficulty tiers against each other over many seeded 2-deck
//! Tractor hands, alternating which partnership is the landlord/defending team to
//! cancel the positional edge, and reports the subject tier's win-rate with a
//! Wilson 95% confidence interval and average point margin. The play/declare/kitty
//! decisions all run through `policy::select_action` exactly as in production, and
//! the honesty boundary is preserved (only Omniscient sees the unredacted state;
//! Grandmaster only ever samples hidden worlds).
//!
//! Games are sharded across worker threads for speed (the bot decision is a pure
//! function of the seeded redacted state — no shared mutable state).
//!
//! Run with:
//!   cargo run --release --example gm_benchmark -- [num_games] [base_seed] [pairs]
//!
//! `pairs` is a comma-separated list of `SUBJECT-OPPONENT` tier names, e.g.
//!   gm-enoch,gm-expert,gm-omni,enoch-expert,omni-enoch
//! Default: `gm-enoch` (plus `enoch-expert` and `omni-enoch` as harness sanity
//! checks when `pairs` is omitted). Tier names: easy, expert, enoch, gm, omni.
//!
//! The Grandmaster search shape is env-tunable for sweeps (see `core/src/bot/policy.rs`):
//!   GM_WORLDS, GM_CANDS, GM_ROLLOUT (0 = roll each world to the last card),
//!   GM_ENOCH (use the Enoch playbook policy — also enables perfect-memory
//!   determinization), GM_BUDGET_MULT (search-budget multiplier vs other tiers).
//!   Base search budget per decision: SHENGJI_BOT_BUDGET_MS (this example
//!   defaults it to 100ms; Grandmaster gets that times GM_BUDGET_MULT).

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
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

fn tier_from_name(s: &str) -> Option<BotDifficulty> {
    match s.trim().to_ascii_lowercase().as_str() {
        "easy" => Some(BotDifficulty::Easy),
        "expert" => Some(BotDifficulty::Expert),
        "enoch" => Some(BotDifficulty::Enoch),
        "gm" | "grandmaster" => Some(BotDifficulty::Grandmaster),
        "omni" | "omniscient" => Some(BotDifficulty::Omniscient),
        _ => None,
    }
}

/// The outcome of one finished hand, from the subject tier's perspective.
struct GameOutcome {
    a_won: bool,
    a_point_margin: isize,
}

/// Build a fully-seeded 4-player, 2-deck Tractor Draw phase, seat 0 the landlord.
/// (Verbatim from `tournament::seeded_draw_phase`.)
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
/// boundary (only Omniscient sees the unredacted state).
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

/// Drive one seeded hand: tier `a` (subject) vs tier `b`. `a_is_landlord_team`
/// selects which partnership plays tier `a`. Result is from `a`'s perspective.
/// (Verbatim driver from `tournament::play_one_hand`.)
fn play_one_hand(
    a_is_landlord_team: bool,
    a: BotDifficulty,
    b: BotDifficulty,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    let tier_of = |seat_idx: usize| -> BotDifficulty {
        let is_landlord_team = seat_idx.is_multiple_of(2);
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

#[derive(Default, Clone, Copy)]
struct Tally {
    completed: usize,
    a_wins: usize,
    a_margin: isize,
    a_landlord_games: usize,
    a_landlord_wins: usize,
    a_attacker_games: usize,
    a_attacker_wins: usize,
}

impl Tally {
    fn merge(&mut self, o: &Tally) {
        self.completed += o.completed;
        self.a_wins += o.a_wins;
        self.a_margin += o.a_margin;
        self.a_landlord_games += o.a_landlord_games;
        self.a_landlord_wins += o.a_landlord_wins;
        self.a_attacker_games += o.a_attacker_games;
        self.a_attacker_wins += o.a_attacker_wins;
    }
}

/// Wilson score interval (95%) for a binomial proportion.
fn wilson_ci(wins: usize, n: usize) -> (f64, f64, f64) {
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let nf = n as f64;
    let p = wins as f64 / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (p + z2 / (2.0 * nf)) / denom;
    let half = (z * ((p * (1.0 - p) / nf) + z2 / (4.0 * nf * nf)).sqrt()) / denom;
    (p * 100.0, (center - half) * 100.0, (center + half) * 100.0)
}

fn run_pair(a: BotDifficulty, b: BotDifficulty, num_games: usize, base_seed: u64, threads: usize) {
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let mut partials: Vec<Tally> = vec![Tally::default(); threads];

    // Independent games, alternating which partnership is the landlord/defending
    // team (g % 2) to cancel positional bias. This is the methodology with a clean
    // ~50% control (a tier configured identically to its opponent scores 50%).
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let next = &next;
            handles.push(scope.spawn(move || {
                let mut t = Tally::default();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= num_games {
                        break;
                    }
                    let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
                    let a_is_landlord_team = g.is_multiple_of(2);
                    if let Some(o) = play_one_hand(a_is_landlord_team, a, b, &mut rng) {
                        t.completed += 1;
                        if o.a_won {
                            t.a_wins += 1;
                        }
                        t.a_margin += o.a_point_margin;
                        if a_is_landlord_team {
                            t.a_landlord_games += 1;
                            if o.a_won {
                                t.a_landlord_wins += 1;
                            }
                        } else {
                            t.a_attacker_games += 1;
                            if o.a_won {
                                t.a_attacker_wins += 1;
                            }
                        }
                    }
                }
                t
            }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            partials[i] = h.join().unwrap();
        }
    });

    let mut tally = Tally::default();
    for p in &partials {
        tally.merge(p);
    }

    let (wr, lo, hi) = wilson_ci(tally.a_wins, tally.completed);
    let secs = start.elapsed().as_secs_f64();
    println!(
        "=== {} (subject) vs {} ({} games) ===",
        a.as_str(),
        b.as_str(),
        tally.completed
    );
    println!(
        "  {} win-rate: {:.2}%   95% CI [{:.2}%, {:.2}%]   ({} wins / {} games)",
        a.as_str(),
        wr,
        lo,
        hi,
        tally.a_wins,
        tally.completed
    );
    let beats = if lo > 50.0 {
        "YES (win-rate CI excludes 50%)"
    } else if hi < 50.0 {
        "NO (win-rate CI below 50%)"
    } else {
        "INCONCLUSIVE on win-rate"
    };
    println!("  beats opponent? {beats}");
    println!(
        "  avg point margin: {:+.2} pts/game",
        tally.a_margin as f64 / tally.completed.max(1) as f64
    );
    if tally.a_landlord_games > 0 {
        println!(
            "    as LANDLORD/defender: {}/{} = {:.1}%",
            tally.a_landlord_wins,
            tally.a_landlord_games,
            tally.a_landlord_wins as f64 / tally.a_landlord_games as f64 * 100.0
        );
    }
    if tally.a_attacker_games > 0 {
        println!(
            "    as ATTACKER:          {}/{} = {:.1}%",
            tally.a_attacker_wins,
            tally.a_attacker_games,
            tally.a_attacker_wins as f64 / tally.a_attacker_games as f64 * 100.0
        );
    }
    println!(
        "  elapsed: {:.1}s  ({:.1} games/s)\n",
        secs,
        tally.completed as f64 / secs.max(1e-9)
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x6A11);
    let pairs_arg = args.get(3).cloned();

    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "100");
    }
    let threads: usize = env::var("GM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let pairs: Vec<(BotDifficulty, BotDifficulty)> = match pairs_arg.as_deref() {
        None => vec![
            (BotDifficulty::Grandmaster, BotDifficulty::Enoch),
            (BotDifficulty::Enoch, BotDifficulty::Expert),
            (BotDifficulty::Omniscient, BotDifficulty::Enoch),
        ],
        Some(spec) => spec
            .split(',')
            .filter_map(|p| {
                let mut it = p.split('-');
                let a = tier_from_name(it.next()?)?;
                let b = tier_from_name(it.next()?)?;
                Some((a, b))
            })
            .collect(),
    };

    println!("GRANDMASTER benchmark");
    println!(
        "games/pair: {num_games}  base_seed: {base_seed:#x}  threads: {threads}  budget_ms: {}",
        env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default()
    );
    for (k, v) in [
        ("GM_WORLDS", "400"),
        ("GM_CANDS", "8"),
        ("GM_ROLLOUT", "0=full"),
        ("GM_ENOCH", "1"),
        ("GM_BUDGET_MULT", "3"),
    ] {
        print!(
            "  {k}={}",
            env::var(k).unwrap_or_else(|_| format!("{v}(default)"))
        );
    }
    println!("\n");

    for (a, b) in pairs {
        run_pair(a, b, num_games, base_seed, threads);
    }
}
