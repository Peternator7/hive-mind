use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;
use std::time::Instant;

use hive_engine::error::HiveError;
use hive_engine::game::GameWinner;
use hive_engine::movement::Move;
use hive_engine::BOARD_SIZE;
use hive_engine::{game::Game, piece::Color};
use hive_ml::encode::translate_game_to_seq_tensor;
use hive_ml::{
    frames::{MultipleGames, SingleGame},
    hypers::{self},
    model::HiveModel,
    translate_to_tensor, PieceEncodable,
};
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{nn, IndexOp, Kind, Tensor};

// input data shape should be [batch, channels, rows, cols]
// output data shape should be [batch, ]

pub fn main() -> Result<(), Box<dyn Error>> {
    let device = tch::Device::cuda_if_available();

    let vs = nn::VarStore::new(device);
    let model = Mutex::new(HiveModel::new(&vs.root()));

    vs.save("models/initial.weights")?;

    let frames = Mutex::new(MultipleGames::default());
    let quantiles = Tensor::from_slice(&[0.5f32, 0.8f32, 0.99f32]);

    for epoch in 1..100 {
        let st = Instant::now();
        let games_played = &AtomicUsize::new(0);
        let games_stalled = &AtomicUsize::new(0);
        let games_lengths: &Mutex<Vec<f32>> = &Mutex::new(Vec::new());
        let games_finished = &AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let handles = (0..hypers::PARALLEL_GAMES)
                .map(|_| {
                    let white_model = &model;
                    let black_model = &model;
                    scope.spawn(|| {
                        let samples = &mut SingleGame::default();
                        assert!(samples.validate_buffers());
                        samples.clear();
                        while frames.lock().unwrap().len() < hypers::TARGET_FRAMES_PER_BATCH {
                            let mut game = Game::new();
                            games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                            let winner = match play_game_to_end(
                                &mut game,
                                white_model,
                                black_model,
                                samples,
                            ) {
                                Ok(winner) => winner,
                                Err(HiveError::TurnLimitHit) => {
                                    samples.clear();
                                    games_stalled
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    continue;
                                }
                                Err(e) => return Err(e),
                            };

                            games_finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            games_lengths.lock().unwrap().push(game.turn() as f32);
                            let mut frames = frames.lock().unwrap();
                            frames.ingest_game(
                                samples,
                                winner,
                                hypers::GAMMA,
                                hypers::LAMBDA,
                                hypers::MAX_FRAMES_PER_GAME,
                            );
                        }

                        Result::<_, HiveError>::Ok(())
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                match handle.join() {
                    Ok(result) => result?,
                    Err(err) => std::panic::resume_unwind(err),
                }
            }

            Result::<(), HiveError>::Ok(())
        })?;

        let mut frames = frames.lock().unwrap();
        let lengths = Tensor::from_slice(games_lengths.lock().unwrap().as_slice());
        let percentiles: Vec<f32> = lengths.quantile(&quantiles, None, false, "linear").try_into()?;

        println!(
            "Total Games: {}, Finished: {}, Stalled: {}, Avg  Time: {}s, Frame Count: {}, P50 Turns: {}, P80 Turns: {}, P99 Turns: {}",
            games_played.load(std::sync::atomic::Ordering::Relaxed),
            games_finished.load(std::sync::atomic::Ordering::Relaxed),
            games_stalled.load(std::sync::atomic::Ordering::Relaxed),
            (Instant::now() - st).as_secs_f32(),
            frames.len(),
            percentiles[0],
            percentiles[1],
            percentiles[2],
        );

        // Do a training loop.
        let st = Instant::now();

        // Create a new optimizer each epoch to avoid momentum carrying over inappropriate
        // from batch to batch.
        let mut adam = nn::Adam::default()
            .build(&vs, hypers::LEARNING_RATE)
            .unwrap();
        _train_loop(&mut adam, &mut *model.lock().unwrap(), &*frames);
        println!(
            "Epoch: {}, Training Iteration Complete, Time Taken: {}s",
            epoch,
            (Instant::now() - st).as_secs_f32()
        );

        frames.clear();

        vs.save(format!("models/epoch_{0}", epoch))?;

        if epoch % 10 == 0 {
            println!("Test against random model...");

            let mut old_vs = VarStore::new(device);
            let old_model = Mutex::new(HiveModel::new(&old_vs.root()));
            old_vs.load("models/initial.weights")?;

            let st = Instant::now();
            let games_played = &AtomicUsize::new(0);
            let white_won = &AtomicUsize::new(0);
            let black_won = &AtomicUsize::new(0);
            let drawn = &AtomicUsize::new(0);

            std::thread::scope(|scope| {
                let handles = (0..hypers::PARALLEL_GAMES)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut last_iter_failed = false;
                            while last_iter_failed
                                || games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                    < hypers::GAMES_PER_AI_SIMULATION
                            {
                                last_iter_failed = false;
                                let mut game = Game::new();
                                let samples = &mut Default::default();
                                match play_game_to_end(&mut game, &model, &old_model, samples) {
                                    Ok(None) => {
                                        drawn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    Ok(Some(Color::White)) => {
                                        white_won
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    Ok(Some(Color::Black)) => {
                                        black_won
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    Err(HiveError::TurnLimitHit) => {
                                        last_iter_failed = true;
                                        continue;
                                    }
                                    Err(e) => return Err(e),
                                }
                            }

                            Result::<_, HiveError>::Ok(())
                        })
                    })
                    .collect::<Vec<_>>();

                for handle in handles {
                    match handle.join() {
                        Ok(result) => result?,
                        Err(err) => std::panic::resume_unwind(err),
                    }
                }

                Result::<(), HiveError>::Ok(())
            })?;

            println!(
                "Vs Random, Wins (new): {}, Wins (random): {}, Ties: {}, Time: {}s",
                white_won.load(std::sync::atomic::Ordering::Relaxed),
                black_won.load(std::sync::atomic::Ordering::Relaxed),
                drawn.load(std::sync::atomic::Ordering::Relaxed),
                (Instant::now() - st).as_secs_f32(),
            );
        }
    }

    Ok(())
}

fn play_game_to_end(
    g: &mut Game,
    white_model: &Mutex<HiveModel>,
    black_model: &Mutex<HiveModel>,
    samples: &mut SingleGame,
) -> hive_engine::Result<Option<Color>> {
    let mut consecutive_passes = 0;
    let mut valid_moves = Vec::new();
    let mut invalid_moves_mask = vec![true; hypers::OUTPUT_LENGTH];
    let mut winner = None;
    loop {
        let model = if g.to_play() == Color::White {
            white_model
        } else {
            black_model
        };

        if g.turn() == hypers::MAX_TURNS_PER_GAME {
            return Err(HiveError::TurnLimitHit);
        }

        if samples.game_state.len() > 2 * hypers::MAX_TURNS_PER_GAME {
            panic!("{}", g.turn());
        }

        let device = model.lock().unwrap().device;

        if let Some(result) = g.is_game_is_over() {
            if let GameWinner::Winner(color) = result {
                winner = Some(color);
            }

            break;
        }

        valid_moves.clear();
        invalid_moves_mask.fill(true);

        g.load_all_potential_moves(&mut valid_moves)?;

        if valid_moves.is_empty() {
            g.make_move(Move::Pass)?;
            consecutive_passes += 1;
            if consecutive_passes >= 6 {
                break;
            }

            continue;
        } else {
            consecutive_passes = 0;
        }

        let playing = g.to_play();
        let _ = translate_game_to_seq_tensor(&g, playing);
        let curr_state = translate_to_tensor(&g, playing);
        let curr_state_batch = curr_state
            .view((
                1,
                hypers::INPUT_ENCODED_DIMS as i64,
                BOARD_SIZE as i64,
                BOARD_SIZE as i64,
            ))
            .to(device);

        let mut map = HashMap::new();

        for mv in &valid_moves {
            let (plane, pos) = match mv {
                Move::MovePiece { piece, to, .. } => (piece.encode(playing), to),
                Move::PlacePiece { piece, position } => (piece.encode(playing), position),
                Move::Pass => unreachable!(),
            };

            let idx = plane * (BOARD_SIZE as usize * BOARD_SIZE as usize)
                + (BOARD_SIZE as usize * pos.0 as usize)
                + (pos.1 as usize);

            // This is a valid move.
            invalid_moves_mask[idx] = false;
            map.entry(idx).insert_entry(mv);
        }

        let invalid_moves_tensor = Tensor::from_slice(&invalid_moves_mask);

        let (value, mut policy) =
            tch::no_grad(|| model.lock().unwrap().value_policy(&curr_state_batch));

        let _ = policy.masked_fill_(&invalid_moves_tensor.to(device), f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");
        g.make_move(**mv)?;

        samples.playing.push(playing);
        samples.game_state.push(curr_state);
        samples.invalid_move_mask.push(invalid_moves_tensor);
        samples.value.push(value.view(1i64).to(tch::Device::Cpu));
        samples
            .selected_policy
            .push(sampled_action_idx.view(1i64).to(tch::Device::Cpu));
    }

    Ok(winner)
}

fn _train_loop(adam: &mut Optimizer, model: &mut HiveModel, frames: &MultipleGames) {
    let state_buffer = Tensor::stack(frames.game_state.as_slice(), 0).to(model.device);
    let mask_buffer = Tensor::stack(frames.invalid_move_mask.as_slice(), 0).to(model.device);
    let selections_buffer = Tensor::stack(frames.selected_policy.as_slice(), 0).to(model.device);
    let gae_buffer = Tensor::stack(frames.gae.as_slice(), 0).to(model.device);
    let target_value_buffer = Tensor::stack(frames.target_value.as_slice(), 0).to(model.device);

    let logp_old = tch::no_grad(|| model.policy(&state_buffer))
        .masked_fill_(&mask_buffer, f64::NEG_INFINITY)
        .log_softmax(1, None)
        .gather(1, &selections_buffer, false);

    let clip_ratio = 0.1f64;

    let mut batch_count = 0;
    let mut total_value_loss = Tensor::zeros(1, (Kind::Float, model.device));
    let mut total_value_count = 0;
    for _ in 0..hypers::TRAIN_ITERS_PER_BATCH {
        batch_count += 1;
        let perms = Tensor::randperm(frames.len() as i64, (Kind::Int64, model.device));
        let mut should_stop_early = false;

        for start in (0..frames.len()).step_by(hypers::BATCH_SIZE) {
            adam.zero_grad();
            let upper = frames.len().min(start + hypers::BATCH_SIZE) as i64;
            let idxs = perms.i((start as i64)..upper);

            let states = state_buffer.index_select(0, &idxs);
            let (values, mut policies) = model.value_policy(&states);

            let adv = gae_buffer.index_select(0, &idxs);

            let logp = policies
                .masked_fill_(&mask_buffer.index_select(0, &idxs), f64::NEG_INFINITY)
                .log_softmax(1, None)
                .gather(1, &selections_buffer.index_select(0, &idxs), false);

            let logp_old = logp_old.index_select(0, &idxs);

            let ratio = (&logp - &logp_old).exp_();
            let clip_adv = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * &adv;
            let pi_loss = (ratio * adv).minimum(&clip_adv).mean(None).neg();

            let approx_kl: f32 = (logp_old - logp).mean(None).try_into().expect("success");

            if approx_kl > hypers::CUTOFF_KL {
                should_stop_early = true;
                break;
            }

            let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                .square()
                .mean(None);

            total_value_loss = tch::no_grad(|| total_value_loss + &value_loss);
            total_value_count += 1;
            let loss: Tensor = value_loss + (hypers::PI_LOSS_RATIO * pi_loss);

            loss.backward();
            adam.step();
        }

        if should_stop_early {
            println!("Ended training after {} iters", batch_count);
            break;
        }
    }

    (total_value_loss / total_value_count).print();
    adam.zero_grad();
}
