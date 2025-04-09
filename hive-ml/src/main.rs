use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use hive_engine::error::HiveError;
use hive_engine::game::GameWinner;
use hive_engine::movement::Move;
use hive_engine::piece::{Insect, Piece};
use hive_engine::position::Position;
use hive_engine::{game::Game, piece::Color};
use hive_ml::encode::{translate_game_to_conv_tensor, translate_to_valid_moves_mask};
use hive_ml::metrics;
use hive_ml::{
    frames::{MultipleGames, SingleGame},
    hypers::{self},
    model::HiveModel,
};
use rand::prelude::Distribution;
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{nn, IndexOp, Kind, Tensor};

// input data shape should be [batch, channels, rows, cols]
// output data shape should be [batch, ]

pub fn main() -> Result<(), Box<dyn Error>> {
    let device = tch::Device::cuda_if_available();
    let provider = metrics::init_meter_provider();
    metrics::record_training_start_time();
    metrics::record_training_status(true);

    let vs = nn::VarStore::new(device);
    let model = Mutex::new(HiveModel::new(&vs.root()));
    // vs.load("models/epoch_120")?;

    let opponent = &model;
    let mut opponent_vs = nn::VarStore::new(device);
    // let opponent = Mutex::new(HiveModel::new(&opponent_vs.root()));
    // opponent_vs.copy(&vs)?;

    let mut lr = hypers::INITIAL_LEARNING_RATE;
    let mut entropy_loss_factor = hypers::ENTROPY_LOSS_RATIO;
    let mut consecutive_epochs_with_less_than_05_training_steps = 0;
    let mut consecutive_epochs_with_more_than_20_training_steps = 0;
    let max_frames_per_game = hypers::MAX_FRAMES_PER_GAME;

    vs.save("models/epoch_0")?;

    let frames = Mutex::new(MultipleGames::default());
    let quantiles = Tensor::from_slice(&[0.5f32, 0.8f32, 0.99f32]);

    // When the win rate gets above 75%, we switch sides and pin the opponent to
    // the current version to learn to beat our previous strategy.
    // let mut model_plays_as = Color::White;
    let mut rolling_win_rate = 0.50;

    println!("Starting train loop");
    for epoch in 1..251 {
        metrics::record_epoch(epoch);

        let st = Instant::now();
        let games_played = &AtomicUsize::new(0);
        let games_won = &AtomicUsize::new(0);
        let games_won_by_model = &AtomicUsize::new(0);
        let games_played_white_model = &AtomicUsize::new(0);
        let games_won_white_model = &AtomicUsize::new(0);
        let games_played_black_model = &AtomicUsize::new(0);
        let games_won_black_model = &AtomicUsize::new(0);
        let games_lengths: &Mutex<Vec<f32>> = &Mutex::new(Vec::new());
        let games_finished = &AtomicUsize::new(0);
        let distribution = rand::distributions::Bernoulli::new(1.0 - rolling_win_rate).unwrap();

        std::thread::scope(|scope| {
            let handles = (0..hypers::PARALLEL_GAMES)
                .map(|_| {
                    scope.spawn(|| {
                        let mut rng = rand::thread_rng();
                        let samples = &mut SingleGame::default();
                        assert!(samples.validate_buffers());
                        samples.clear();

                        while frames.lock().unwrap().len() < hypers::TARGET_FRAMES_PER_BATCH {
                            let model_plays_as;
                            let white_model;
                            let black_model;
                            if distribution.sample(&mut rng) {
                                model_plays_as = Color::White;
                                white_model = &model;
                                black_model = opponent;
                                games_played_white_model.fetch_add(1, Ordering::Relaxed);
                            } else {
                                model_plays_as = Color::Black;
                                white_model = opponent;
                                black_model = &model;
                                games_played_black_model.fetch_add(1, Ordering::Relaxed);
                            };

                            let mut game = Game::new();
                            let game_start_time = Instant::now();
                            games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            metrics::increment_games_played(model_plays_as);

                            let winner = match play_game_to_end(
                                &mut game,
                                model_plays_as,
                                white_model,
                                black_model,
                                samples,
                            ) {
                                Ok(winner) => winner,
                                Err(HiveError::TurnLimitHit) => {
                                    samples.clear();
                                    continue;
                                }
                                Err(e) => return Err(e),
                            };

                            let elapsed = Instant::now() - game_start_time;
                            metrics::record_game_duration(elapsed.as_secs_f64(), model_plays_as);

                            if winner == Some(Color::White) {
                                games_won.fetch_add(1, Ordering::Relaxed);
                            }

                            if winner == Some(model_plays_as) {
                                games_won_by_model.fetch_add(1, Ordering::Relaxed);
                                metrics::increment_model_won(model_plays_as);

                                if model_plays_as == Color::White {
                                    games_won_white_model.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    games_won_black_model.fetch_add(1, Ordering::Relaxed);
                                }
                            }

                            metrics::increment_games_finished(model_plays_as, winner);
                            games_finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                            metrics::record_game_turns(game.turn(), model_plays_as);
                            games_lengths.lock().unwrap().push(game.turn() as f32);

                            let mut frames = frames.lock().unwrap();
                            frames.ingest_game(
                                samples,
                                winner,
                                hypers::GAMMA,
                                hypers::LAMBDA,
                                max_frames_per_game,
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
        let percentiles: Vec<f32> = lengths
            .quantile(&quantiles, None, false, "linear")
            .try_into()?;

        let win_rate = games_won.load(std::sync::atomic::Ordering::Relaxed) as f64
            / games_finished.load(std::sync::atomic::Ordering::Relaxed) as f64;

        let model_win_rate = games_won_by_model.load(std::sync::atomic::Ordering::Relaxed) as f64
            / games_finished.load(std::sync::atomic::Ordering::Relaxed) as f64;

        // if model_plays_as == Color::Black {
        //     win_rate = 100.0 - win_rate;
        // }

        println!(
            "Epoch: {}, Played: {}, Finished: {}, WR: {:.2}, MWR: {:.2}, Time: {:.2}s, Frames: {}, P50 Turns: {:.0}, P80 Turns: {:.0}, P99 Turns: {:.0}",
            epoch,
            games_played.load(std::sync::atomic::Ordering::Relaxed),
            games_finished.load(std::sync::atomic::Ordering::Relaxed),
            100.0 * win_rate,
            100.0 * model_win_rate,
            (Instant::now() - st).as_secs_f32(),
            frames.len(),
            percentiles[0],
            percentiles[1],
            percentiles[2],
        );

        metrics::record_data_generation_duration((Instant::now() - st).as_secs_f64());

        let white_win_rate = games_won_white_model.load(Ordering::Relaxed) as f64
            / games_played_white_model.load(Ordering::Relaxed) as f64;

        let black_win_rate = games_won_black_model.load(Ordering::Relaxed) as f64
            / games_played_black_model.load(Ordering::Relaxed) as f64;

        // The model can learn a strategy that works better from one side or the other.
        // Since white goes first, it's typically from the white side so we use the ratio between the win
        // rate of white and black to decide which color we should play more of.
        let modified_win_rate = white_win_rate / (white_win_rate + black_win_rate);

        rolling_win_rate = hypers::WIN_RATE_SMOOTHING_FACTOR * rolling_win_rate
            + (1.0 - hypers::WIN_RATE_SMOOTHING_FACTOR) * modified_win_rate;

        if (rolling_win_rate - 0.5).abs() > 0.05 {
            entropy_loss_factor *= 1.10;
            entropy_loss_factor = entropy_loss_factor.min(hypers::MAX_ENTROPY_LOSS_RATIO);
        } else {
            entropy_loss_factor *= 0.975;
            entropy_loss_factor = entropy_loss_factor.max(hypers::MIN_ENTROPY_LOSS_RATIO);
        }

        // Do a training loop.
        let st = Instant::now();

        // Create a new optimizer each epoch to avoid momentum carrying over inappropriate
        // from batch to batch.
        let mut adam = nn::Adam::default().build(&vs, lr).unwrap();
        metrics::record_learning_rate(lr);
        metrics::record_entropy_loss_scale(entropy_loss_factor);

        let train_steps = _train_loop(
            &mut adam,
            &mut *model.lock().unwrap(),
            &*frames,
            entropy_loss_factor,
        );

        if train_steps < 5 {
            consecutive_epochs_with_less_than_05_training_steps += 1;
            consecutive_epochs_with_more_than_20_training_steps = 0;
        } else if train_steps > 20 {
            consecutive_epochs_with_less_than_05_training_steps = 0;
            consecutive_epochs_with_more_than_20_training_steps += 1;
        } else {
            consecutive_epochs_with_less_than_05_training_steps = 0;
            consecutive_epochs_with_more_than_20_training_steps = 0;
        }

        // if consecutive_epochs_with_less_than_05_training_steps > 5 {
        //     consecutive_epochs_with_less_than_05_training_steps = 0;
        //     let old_lr = lr;
        //     lr = hypers::MIN_LEARNING_RATE.max(lr * 0.800);
        //     println!(
        //         "Decreased Learning Rate, Old: {:.7}, New: {:.7}",
        //         old_lr, lr
        //     );
        // } else if consecutive_epochs_with_more_than_20_training_steps > 5 {
        //     consecutive_epochs_with_more_than_20_training_steps = 0;
        //     let old_lr = lr;
        //     lr = hypers::INITIAL_LEARNING_RATE.min(lr * 1.1);
        //     println!(
        //         "Increased Learning Rate, Old: {:.7}, New: {:.7}",
        //         old_lr, lr
        //     );
        // }

        metrics::record_training_duration((Instant::now() - st).as_secs_f64());
        println!(
            "Epoch: {}, Training Iteration Complete, Time Taken: {}s",
            epoch,
            (Instant::now() - st).as_secs_f32()
        );

        frames.clear();

        vs.save(format!("models/epoch_{0}", epoch))?;

        if epoch % 10 == 0 {
            // println!("Updating opponent to current model");
            // opponent_vs.copy(&vs)?;
            // metrics::increment_leveled_up_opponent();

            lr = hypers::MIN_LEARNING_RATE.max(lr * 0.800);

            println!("Testing against initial model...");
            let mut old_vs = VarStore::new(device);
            let old_model = Mutex::new(HiveModel::new(&old_vs.root()));
            old_vs.load("models/epoch_0")?;

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
                                match play_game_to_end(
                                    &mut game,
                                    Color::White,
                                    &model,
                                    &old_model,
                                    samples,
                                ) {
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

            let white_won = white_won.load(Ordering::Relaxed);
            let black_won = black_won.load(Ordering::Relaxed);

            metrics::record_win_rate_vs_initial(
                100.0 * white_won as f64 / (white_won + black_won) as f64,
            );

            println!(
                "Vs Random, Wins (new): {}, Wins (random): {}, Ties: {}, Time: {:.2}s",
                white_won,
                black_won,
                drawn.load(std::sync::atomic::Ordering::Relaxed),
                (Instant::now() - st).as_secs_f32(),
            );
        }
    }

    metrics::record_training_status(false);
    provider.shutdown()?;
    Ok(())
}

fn play_game_to_end(
    g: &mut Game,
    training_model_color: Color,
    white_model: &Mutex<HiveModel>,
    black_model: &Mutex<HiveModel>,
    samples: &mut SingleGame,
) -> hive_engine::Result<Option<Color>> {
    let mut consecutive_passes = 0;
    let mut valid_moves = Vec::new();
    let mut invalid_moves_mask = vec![true; hypers::OUTPUT_LENGTH];
    let mut winner = None;

    // Since the model plays pieces relative to other pieces, it doesn't know how to make
    // the first move.
    g.make_move(Move::PlacePiece {
        piece: Piece {
            role: Insect::Grasshopper,
            color: Color::White,
            id: 0,
        },
        position: Position(16, 16),
    })?;

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

        let playing = g.to_play();

        g.load_all_potential_moves(&mut valid_moves)?;

        if valid_moves.is_empty() {
            if playing == training_model_color {
                metrics::increment_move_made(Move::Pass, training_model_color);
            }

            g.make_move(Move::Pass)?;
            consecutive_passes += 1;
            if consecutive_passes >= 6 {
                break;
            }

            continue;
        } else {
            consecutive_passes = 0;
        }

        let curr_state = translate_game_to_conv_tensor(&g, playing);
        let curr_state_batch = curr_state.unsqueeze(0).to(device);

        let map = translate_to_valid_moves_mask(&g, &valid_moves, playing, &mut invalid_moves_mask);
        let invalid_moves_tensor = Tensor::from_slice(&invalid_moves_mask);

        let (value, mut policy) =
            tch::no_grad(|| model.lock().unwrap().value_policy(&curr_state_batch));

        let _ = policy.masked_fill_(&invalid_moves_tensor.to(device), f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");
        g.make_move(*mv)?;

        if playing == training_model_color {
            samples.playing.push(playing);
            samples.game_state.push(curr_state);
            samples.invalid_move_mask.push(invalid_moves_tensor);
            samples.value.push(value.view(1i64).to(tch::Device::Cpu));
            samples
                .selected_policy
                .push(sampled_action_idx.view(1i64).to(tch::Device::Cpu));

            metrics::increment_move_made(*mv, training_model_color);
        }
    }

    Ok(winner)
}

fn _train_loop(
    adam: &mut Optimizer,
    model: &mut HiveModel,
    frames: &MultipleGames,
    entropy_loss_scaling: f64,
) -> usize {
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

    model.set_train_mode(true);
    for _ in 0..hypers::TRAIN_ITERS_PER_BATCH {
        batch_count += 1;
        let perms = Tensor::randperm(frames.len() as i64, (Kind::Int64, model.device));
        let mut should_stop_early = false;

        for start in (0..frames.len()).step_by(hypers::BATCH_SIZE) {
            adam.zero_grad();
            let upper = frames.len().min(start + hypers::BATCH_SIZE) as i64;
            let idxs = perms.i((start as i64)..upper);

            let states = state_buffer.index_select(0, &idxs);
            let mask = &mask_buffer.index_select(0, &idxs);

            let (values, mut policies) = model.value_policy(&states);
            let _ = policies.masked_fill_(&mask, f64::NEG_INFINITY);

            let adv = gae_buffer.index_select(0, &idxs);

            let logp = policies.log_softmax(1, None).gather(
                1,
                &selections_buffer.index_select(0, &idxs),
                false,
            );

            let logp_old = logp_old.index_select(0, &idxs);

            let ratio = (&logp - &logp_old).exp_();
            let clip_adv = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * &adv;

            let pi_loss = (ratio * adv).minimum(&clip_adv).mean(None);

            let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                .square()
                .mean(None);

            total_value_loss = tch::no_grad(|| total_value_loss + &value_loss);
            total_value_count += 1;

            let p_prob = policies.softmax(1, None) + 0.00001;
            let h = mask.logical_not() * (&p_prob * p_prob.log());
            let entropy_loss = h.sum_dim_intlist(1, false, None).neg().mean(None);

            // Tensor::stack(&[&value_loss, &pi_loss, &entropy_loss], 0).print();

            metrics::record_minibatch_statistics(
                f64::try_from(&value_loss).unwrap(),
                f64::try_from(&pi_loss).unwrap(),
                f64::try_from(&entropy_loss).unwrap(),
            );

            let loss: Tensor = value_loss
                - (hypers::PI_LOSS_RATIO * pi_loss)
                - (entropy_loss_scaling * entropy_loss);

            let approx_kl: f32 = (logp_old - logp).mean(None).try_into().expect("success");

            if batch_count > 1 && approx_kl > hypers::CUTOFF_KL {
                should_stop_early = true;
                break;
            }

            loss.backward();
            adam.step();
        }

        if should_stop_early {
            println!("Ended training after {} iters", batch_count);
            break;
        }
    }

    model.set_train_mode(false);

    let mse = total_value_loss / total_value_count;
    mse.print();
    let mse: f64 = mse.try_into().unwrap();

    metrics::record_value_mse(mse);

    adam.zero_grad();

    metrics::record_training_batches(batch_count);
    batch_count as usize
}
