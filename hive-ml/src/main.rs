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
use hive_ml::encode::{
    self, translate_game_to_conv_tensor, translate_game_to_seq_tensor,
    translate_to_valid_moves_mask,
};
use hive_ml::hypers::SWITCHOVER_EPOCH;
use hive_ml::model::HiveModel;
use hive_ml::model2::HiveTransformerModel;
use hive_ml::{
    frames::{MultipleGames, SingleGame},
    hypers::{self},
};
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{nn, IndexOp, Kind, Tensor};

// input data shape should be [batch, channels, rows, cols]
// output data shape should be [batch, ]

pub fn main() -> Result<(), Box<dyn Error>> {
    let device = tch::Device::cuda_if_available();

    let cnn_vs = nn::VarStore::new(device);
    let cnn_model = Mutex::new(HiveModel::new(&cnn_vs.root()));

    let transformer_vs = nn::VarStore::new(device);
    let transformer_model = Mutex::new(HiveTransformerModel::new(&transformer_vs.root()));

    transformer_vs.save("models/epoch_0_transformer")?;
    cnn_vs.save("models/epoch_0")?;

    let frames = Mutex::new(MultipleGames::default());
    let quantiles = Tensor::from_slice(&[0.5f32, 0.8f32, 0.99f32]);

    for epoch in 1..100 {
        let st = Instant::now();
        let games_played = &AtomicUsize::new(0);
        let games_stalled = &AtomicUsize::new(0);
        let games_lengths: &Mutex<Vec<f32>> = &Mutex::new(Vec::new());
        let games_finished = &AtomicUsize::new(0);

        let model_to_use_for_decision_making = if epoch <= SWITCHOVER_EPOCH {
            Agent::Cnn(&cnn_model)
        } else {
            Agent::Transformer(&transformer_model)
        };

        std::thread::scope(|scope| {
            let handles = (0..hypers::PARALLEL_GAMES)
                .map(|_| {
                    scope.spawn(|| {
                        let samples = &mut SingleGame::default();
                        assert!(samples.validate_buffers());
                        samples.clear();
                        while frames.lock().unwrap().len() < hypers::TARGET_FRAMES_PER_BATCH {
                            let mut game = Game::new();
                            games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                            let winner = match play_game_to_end(
                                &mut game,
                                device,
                                &model_to_use_for_decision_making,
                                &model_to_use_for_decision_making,
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
        let percentiles: Vec<f32> = lengths
            .quantile(&quantiles, None, false, "linear")
            .try_into()?;

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
            .build(&transformer_vs, hypers::LEARNING_RATE)
            .unwrap();

        let mut adam_cnn = nn::Adam::default()
            .build(&cnn_vs, hypers::LEARNING_RATE)
            .unwrap();

        _train_loop(
            &mut adam,
            &mut *transformer_model.lock().unwrap(),
            &mut adam_cnn,
            &mut *cnn_model.lock().unwrap(),
            epoch <= hypers::SWITCHOVER_EPOCH,
            &*frames,
        );

        println!(
            "Epoch: {}, Training Iteration Complete, Time Taken: {}s",
            epoch,
            (Instant::now() - st).as_secs_f32()
        );

        frames.clear();

        transformer_vs.save(format!("models/epoch_{0}_transformer", epoch))?;
        cnn_vs.save(format!("models/epoch_{0}", epoch))?;

        if epoch % 10 == 0 {
            println!("Test against random model...");

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
                                    device,
                                    &Agent::Transformer(&transformer_model),
                                    &Agent::Cnn(&old_model),
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

pub enum Agent<'a> {
    Transformer(&'a Mutex<HiveTransformerModel>),
    Cnn(&'a Mutex<HiveModel>),
}

impl Agent<'_> {
    pub fn model_type(&self) -> &str {
        match self {
            Agent::Transformer(..) => "transformer",
            Agent::Cnn(..) => "cnn",
        }
    }
}

fn play_game_to_end(
    g: &mut Game,
    device: tch::Device,
    white_model: &Agent,
    black_model: &Agent,
    samples: &mut SingleGame,
) -> hive_engine::Result<Option<Color>> {
    let mut consecutive_passes = 0;
    let mut valid_moves = Vec::new();
    let mut invalid_moves_mask = vec![true; hypers::OUTPUT_LENGTH];
    let mut winner = None;
    let lengths_mask = encode::length_masks().copy();

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

        if samples.pieces.len() > 2 * hypers::MAX_TURNS_PER_GAME {
            panic!("{}", g.turn());
        }

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
        let (pieces, locations, seq_length) =
            tch::no_grad(|| translate_game_to_seq_tensor(&g, playing));

        let game_state = tch::no_grad(|| translate_game_to_conv_tensor(&g, playing));
        let game_state_batch = game_state.unsqueeze(0).to(device);

        let pieces_batch = pieces.unsqueeze(0).to(device);
        let location_batch = locations.unsqueeze(0).to(device);
        let curr_state_mask = lengths_mask
            .i(seq_length as i64)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device);

        let map = translate_to_valid_moves_mask(&g, &valid_moves, playing, &mut invalid_moves_mask);
        let invalid_moves_tensor = Tensor::from_slice(&invalid_moves_mask);

        let (value, mut policy) = tch::no_grad(|| match model {
            Agent::Transformer(mutex) => mutex.lock().unwrap().value_policy(
                &pieces_batch,
                &location_batch,
                Some(&curr_state_mask),
            ),
            Agent::Cnn(mutex) => mutex.lock().unwrap().value_policy(&game_state_batch),
        });

        let _ = policy.masked_fill_(&invalid_moves_tensor.to(device), f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");
        g.make_move(*mv)?;

        samples.playing.push(playing);
        samples.game_state.push(game_state);
        samples.locations.push(locations);
        samples.pieces.push(pieces);
        samples.invalid_move_mask.push(invalid_moves_tensor);
        samples.value.push(value.view(1i64).to(tch::Device::Cpu));
        samples
            .selected_policy
            .push(sampled_action_idx.view(1i64).to(tch::Device::Cpu));
        samples
            .seq_length
            .push(Tensor::from(seq_length as i64).view(1i64));
    }

    Ok(winner)
}

fn _train_loop(
    adam: &mut Optimizer,
    model: &mut HiveTransformerModel,
    adam_cnn: &mut Optimizer,
    cnn: &mut HiveModel,
    cnn_is_baseline: bool,
    frames: &MultipleGames,
) {
    let length_masks = encode::length_masks();

    let state_buffer = Tensor::stack(frames.game_state.as_slice(), 0).to(model.device);
    let pieces_buffer = Tensor::stack(frames.pieces.as_slice(), 0).to(model.device);
    let mask_buffer = Tensor::stack(frames.invalid_move_mask.as_slice(), 0).to(model.device);
    let selections_buffer = Tensor::stack(frames.selected_policy.as_slice(), 0).to(model.device);
    let gae_buffer = Tensor::stack(frames.gae.as_slice(), 0).to(model.device);
    let target_value_buffer = Tensor::stack(frames.target_value.as_slice(), 0).to(model.device);

    let location_buffer = Tensor::stack(frames.locations.as_slice(), 0).to(model.device);
    let length_mask_buffer = length_masks
        .index_select(
            0,
            &Tensor::stack(frames.sequence_length.as_slice(), 0).squeeze(),
        )
        .unsqueeze(1)
        .to(model.device);

    // let logp_old =
    //     tch::no_grad(|| model.policy(&pieces_buffer, &location_buffer, Some(&length_mask_buffer)))
    //         .masked_fill_(&mask_buffer, f64::NEG_INFINITY)
    //         .log_softmax(1, None)
    //         .gather(1, &selections_buffer, false);

    let old_cnn_output =
        tch::no_grad(|| cnn.policy(&state_buffer)).masked_fill_(&mask_buffer, f64::NEG_INFINITY);

    let logp_old_cnn = old_cnn_output.log_softmax(1, None);
    //.gather(1, &selections_buffer, false);

    let old_transformer_output =
        tch::no_grad(|| model.policy(&pieces_buffer, &location_buffer, Some(&length_mask_buffer)))
            .masked_fill_(&mask_buffer, f64::NEG_INFINITY);

    let logp_old_transformer = old_transformer_output.log_softmax(1, None);

    let softmax_old_transformer = old_transformer_output.softmax(1, None) + 0.000001;
    let softmax_old_cnn = old_cnn_output.softmax(1, None) + 0.000001;
    let cnn_vs_transformer_policy =
        softmax_old_transformer
            .log()
            .kl_div(&softmax_old_cnn, tch::Reduction::Sum, false);
    //     logp_old_transformer.kl_div(&logp_old_cnn, tch::Reduction::Sum, true);

    println!(
        "KL Divergence Between Transformer Policy and CNN: {}",
        cnn_vs_transformer_policy / softmax_old_transformer.size()[0]
    );

    let logp_old = if cnn_is_baseline {
        logp_old_cnn
    } else {
        logp_old_transformer
    };
    let logp_old = logp_old.gather(1, &selections_buffer, false);
    let clip_ratio = 0.1f64;

    let mut batch_count = 0;
    let mut total_value_loss_transformer = Tensor::zeros(1, (Kind::Float, model.device));
    let mut total_value_loss_cnn = Tensor::zeros(1, (Kind::Float, model.device));
    let mut total_value_count = 0;

    model.train_mode.store(true, Ordering::Relaxed);
    for _ in 0..hypers::TRAIN_ITERS_PER_BATCH {
        batch_count += 1;
        let perms = Tensor::randperm(frames.len() as i64, (Kind::Int64, model.device));
        let mut should_stop_early = false;

        for start in (0..frames.len()).step_by(hypers::BATCH_SIZE) {
            adam.zero_grad();
            adam_cnn.zero_grad();

            let upper = frames.len().min(start + hypers::BATCH_SIZE) as i64;
            let idxs = perms.i((start as i64)..upper);

            let pieces = pieces_buffer.index_select(0, &idxs);
            let locations = location_buffer.index_select(0, &idxs);
            let length_masks = length_mask_buffer.index_select(0, &idxs);
            let adv = gae_buffer.index_select(0, &idxs);
            let logp_old = logp_old.index_select(0, &idxs);

            // Compute the transformer loss
            let (values, mut policies) =
                model.value_policy(&pieces, &locations, Some(&length_masks));

            if true {
                let logp = policies
                    .masked_fill_(&mask_buffer.index_select(0, &idxs), f64::NEG_INFINITY)
                    .log_softmax(1, None)
                    .gather(1, &selections_buffer.index_select(0, &idxs), false);

                let ratio = (&logp - &logp_old).exp_();
                let clip_adv = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * &adv;
                let pi_loss = (ratio * &adv).minimum(&clip_adv).mean(None).neg();

                let approx_kl: f32 = (&logp_old - logp).mean(None).try_into().expect("success");

                if !cnn_is_baseline {
                    if approx_kl > hypers::CUTOFF_KL {
                        should_stop_early = true;
                        break;
                    }
                }

                let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                    .square()
                    .mean(None);

                total_value_loss_transformer =
                    tch::no_grad(|| total_value_loss_transformer + &value_loss);
                total_value_count += 1;
                let loss: Tensor = value_loss + (hypers::PI_LOSS_RATIO * pi_loss);

                loss.backward();
                adam.step();
            } else {
                let policy_softmax = policies.softmax(1, None) + 0.000001;
                let policy_targets = softmax_old_cnn.index_select(0, &idxs);
                let pi_loss =
                    policy_softmax
                        .log()
                        .kl_div(&policy_targets, tch::Reduction::Sum, false)
                        / hypers::BATCH_SIZE as f64;

                let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                    .square()
                    .mean(None);

                total_value_loss_transformer =
                    tch::no_grad(|| total_value_loss_transformer + &value_loss);
                total_value_count += 1;
                let loss = value_loss + pi_loss;

                loss.backward();
                adam.step();
            }

            // Compute the cnn loss
            let states = state_buffer.index_select(0, &idxs);
            let (cnn_values, mut cnn_policies) = cnn.value_policy(&states);

            let logp = cnn_policies
                .masked_fill_(&mask_buffer.index_select(0, &idxs), f64::NEG_INFINITY)
                .log_softmax(1, None)
                .gather(1, &selections_buffer.index_select(0, &idxs), false);

            let ratio = (&logp - &logp_old).exp_();
            let clip_adv = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * &adv;
            let pi_loss_cnn = (ratio * adv).minimum(&clip_adv).mean(None).neg();

            if cnn_is_baseline {
                let approx_kl: f32 = (logp_old - logp).mean(None).try_into().expect("success");

                if approx_kl > hypers::CUTOFF_KL {
                    should_stop_early = true;
                    break;
                }
            }

            let value_loss = (cnn_values - target_value_buffer.index_select(0, &idxs))
                .square()
                .mean(None);

            total_value_loss_cnn = tch::no_grad(|| total_value_loss_cnn + &value_loss);
            total_value_count += 1;
            let cnn_loss: Tensor = value_loss + (hypers::PI_LOSS_RATIO * pi_loss_cnn);

            cnn_loss.backward();
            adam_cnn.step();
        }

        if should_stop_early {
            println!("Ended training after {} iters", batch_count);
            break;
        }
    }

    model.train_mode.store(false, Ordering::Relaxed);
    (total_value_loss_cnn / total_value_count).print();
    (total_value_loss_transformer / total_value_count).print();
    adam.zero_grad();
    adam_cnn.zero_grad();
}
