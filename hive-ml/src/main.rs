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
use rand::seq::{IndexedRandom, SliceRandom};
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{nn, IndexOp, Kind, Tensor};

// input data shape should be [batch, channels, rows, cols]
// output data shape should be [batch, ]

struct ModelData {
    name: String,
    vs: VarStore,
    frames: Mutex<MultipleGames>,
    auxiliary_frames: Mutex<MultipleGames>,
    model: Mutex<HiveModel>,
}

struct PlayerGameData<'a> {
    name: &'a str,
    model: &'a HiveModel,
    samples: &'a mut SingleGame,
}

impl ModelData {
    fn new(
        name: &str,
        device: tch::Device,
        load_from_epoch: Option<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut vs = nn::VarStore::new(device);
        let model = Mutex::new(HiveModel::new(&vs.root()));

        if let Some(epoch) = load_from_epoch {
            vs.load(format!("models/{}_epoch_{}", name, epoch))?;
        }

        Ok(Self {
            name: name.to_string(),
            vs,
            frames: Default::default(),
            auxiliary_frames: Default::default(),
            model,
        })
    }

    fn save_model(&self, epoch: usize) -> Result<(), Box<dyn Error>> {
        self.vs
            .save(format!("models/{}_epoch_{}", self.name, epoch))?;
        Ok(())
    }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let device = tch::Device::cuda_if_available();
    let provider = metrics::init_meter_provider();
    metrics::record_training_start_time();
    metrics::record_training_status(true);

    let random_model = ModelData::new("random_model", device, None)?;

    // Create four models to train in parallel
    let mut models = Vec::new();
    for i in 0..hypers::NUMBER_OF_MODELS {
        models.push(ModelData::new(hypers::MODEL_NAMES[i], device, None)?);
    }

    let indices = (0..models.len()).collect::<Vec<_>>();
    let play_as: &[Color] = &[Color::White, Color::Black];

    // Save initial state of all models
    for model in &models {
        model.save_model(0)?;
    }

    let mut lr = hypers::INITIAL_LEARNING_RATE;
    let entropy_loss_factor = hypers::ENTROPY_LOSS_RATIO;
    let max_frames_per_game = hypers::MAX_FRAMES_PER_GAME;
    let quantiles = Tensor::from_slice(&[0.5f32, 0.8f32, 0.99f32]);

    println!("Starting train loop");
    let mut main_rng = rand::thread_rng();
    for epoch in 1..hypers::EPOCHS {
        metrics::record_epoch(epoch);
        models.shuffle(&mut main_rng);

        // Train each model in sequence
        for model_data in models.iter() {
            println!("Training model {}", model_data.name);

            let st = Instant::now();
            let games_played = &AtomicUsize::new(0);
            let games_lengths: &Mutex<Vec<f32>> = &Mutex::new(Vec::new());
            let games_finished = &AtomicUsize::new(0);

            std::thread::scope(|scope| {
                let handles = (0..hypers::PARALLEL_GAMES)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut thread_vs = nn::VarStore::new(device);
                            let thread_model = &HiveModel::new(&thread_vs.root());
                            thread_vs.copy(&model_data.vs).unwrap();

                            let mut opponents = Vec::new();
                            for model_data in models.iter() {
                                let mut opponent_vs = nn::VarStore::new(device);
                                let opponent = HiveModel::new(&opponent_vs.root());
                                opponent_vs.copy(&model_data.vs).unwrap();
                                opponents.push(opponent);
                            }

                            let mut rng = rand::thread_rng();
                            let samples = &mut SingleGame::default();
                            let opponent_samples = &mut SingleGame::default();

                            while model_data.frames.lock().unwrap().len()
                                < hypers::TARGET_FRAMES_PER_BATCH
                            {
                                samples.clear();
                                opponent_samples.clear();

                                // The rest of the game playing code remains the same
                                let model_plays_as = *play_as.choose(&mut rng).unwrap();
                                let mut white_model;
                                let mut black_model;

                                let opponent_idx = *indices.choose(&mut rng).unwrap();
                                let opponent = &opponents[opponent_idx];
                                let opponent_name = &*models[opponent_idx].name;

                                samples.playing = Some(model_plays_as);
                                opponent_samples.playing = Some(model_plays_as.opposing());
                                if model_plays_as == Color::White {
                                    white_model = PlayerGameData {
                                        name: model_data.name.as_str(),
                                        samples: samples,
                                        model: thread_model,
                                    };

                                    black_model = PlayerGameData {
                                        name: opponent_name,
                                        samples: opponent_samples,
                                        model: opponent,
                                    };
                                } else {
                                    black_model = PlayerGameData {
                                        name: model_data.name.as_str(),
                                        samples: samples,
                                        model: thread_model,
                                    };

                                    white_model = PlayerGameData {
                                        name: opponent_name,
                                        samples: opponent_samples,
                                        model: opponent,
                                    };
                                };

                                let mut game = Game::new();
                                let game_start_time = Instant::now();
                                games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                let mut stalled = false;
                                let winner = match play_game_to_end(
                                    &mut game,
                                    &mut white_model,
                                    &mut black_model,
                                ) {
                                    Ok(winner) => winner,
                                    Err(HiveError::TurnLimitHit) => {
                                        stalled = true;
                                        None
                                    }
                                    Err(e) => return Err(e),
                                };

                                let elapsed = Instant::now() - game_start_time;
                                metrics::record_game_duration(
                                    elapsed.as_secs_f64(),
                                    model_plays_as,
                                    &model_data.name,
                                    opponent_name,
                                );

                                metrics::increment_games_finished(
                                    model_plays_as,
                                    winner,
                                    stalled,
                                    &model_data.name,
                                    opponent_name,
                                );

                                games_finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                metrics::record_game_turns(
                                    game.turn(),
                                    model_plays_as,
                                    &model_data.name,
                                    opponent_name,
                                );

                                games_lengths.lock().unwrap().push(game.turn() as f32);

                                let mut frames = model_data.frames.lock().unwrap();
                                frames.ingest_game(
                                    samples,
                                    winner,
                                    stalled,
                                    hypers::GAMMA,
                                    hypers::LAMBDA,
                                    max_frames_per_game,
                                );

                                metrics::record_frame_buffer_count(frames.len());
                                drop(frames);

                                let mut frames = models[opponent_idx].frames.lock().unwrap();
                                if frames.len() < hypers::TARGET_FRAMES_PER_BATCH {
                                    frames.ingest_game(
                                        opponent_samples,
                                        winner,
                                        stalled,
                                        hypers::GAMMA,
                                        hypers::LAMBDA,
                                        max_frames_per_game,
                                    );
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

            let frames = model_data.frames.lock().unwrap();
            let lengths = Tensor::from_slice(games_lengths.lock().unwrap().as_slice());
            let percentiles: Vec<f32> = lengths
                .quantile(&quantiles, None, false, "linear")
                .try_into()?;

            println!(
                "Epoch: {}, Model: {}, Played: {}, Finished: {}, Time: {:.2}s, Frames: {}, P50 Turns: {:.0}, P80 Turns: {:.0}, P99 Turns: {:.0}",
                epoch,
                model_data.name,
                games_played.load(std::sync::atomic::Ordering::Relaxed),
                games_finished.load(std::sync::atomic::Ordering::Relaxed),
                (Instant::now() - st).as_secs_f32(),
                frames.len(),
                percentiles[0],
                percentiles[1],
                percentiles[2],
            );

            metrics::record_data_generation_duration((Instant::now() - st).as_secs_f64());
        }

        let st = Instant::now();

        for model_data in models.iter_mut() {
            let mut adam = nn::Adam::default().build(&model_data.vs, lr).unwrap();
            let frames = &mut *model_data.frames.lock().unwrap();
            metrics::record_learning_rate(lr);

            _train_policy_phase(
                &mut adam,
                &mut *model_data.model.lock().unwrap(),
                frames,
                entropy_loss_factor,
                &model_data.name,
            );

            model_data.save_model(epoch)?;
            metrics::record_training_duration((Instant::now() - st).as_secs_f64());
            metrics::record_frame_buffer_count(frames.len());
        }

        for model_data in models.iter_mut() {
            println!("Copying frames into auxiliary.");
            let curr_batch = &mut *model_data.frames.lock().unwrap();
            let combined = &mut *model_data.auxiliary_frames.lock().unwrap();
            combined.ingest_multiple_games(curr_batch, hypers::AUXILIARY_PHASE_TRAIN_FREQUENCY);

            if epoch % hypers::AUXILIARY_PHASE_TRAIN_FREQUENCY == 0 {
                let mut adam = nn::Adam::default().build(&model_data.vs, lr).unwrap();
                metrics::record_learning_rate(lr);
                println!(
                    "Running Auxiliary Phase training, frames: {}",
                    combined.len()
                );

                _train_auxiliary_phase(
                    &mut adam,
                    &mut *model_data.model.lock().unwrap(),
                    combined,
                    &model_data.name,
                );

                combined.clear();
            }
        }

        // lr = hypers::MIN_LEARNING_RATE.max(lr * hypers::LEARNING_RATE_DECAY);
        lr -= hypers::LEARNING_RATE_DECREASE;

        if epoch % 10 == 0 {
            for model_data in models.iter() {
                println!("Testing {} against initial model...", model_data.name);

                let st = Instant::now();
                let games_played = &AtomicUsize::new(0);
                let white_won = &AtomicUsize::new(0);
                let black_won = &AtomicUsize::new(0);
                let drawn = &AtomicUsize::new(0);

                std::thread::scope(|scope| {
                    let handles = (0..hypers::PARALLEL_GAMES)
                        .map(|_| {
                            scope.spawn(|| {
                                let mut thread_vs = nn::VarStore::new(device);
                                let thread_model = &HiveModel::new(&thread_vs.root());
                                thread_vs.copy(&model_data.vs).unwrap();

                                let mut old_vs = VarStore::new(device);
                                let old_model = HiveModel::new(&old_vs.root());
                                old_vs.copy(&random_model.vs).unwrap();

                                let mut last_iter_failed = false;
                                while last_iter_failed
                                    || games_played
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                        < hypers::GAMES_PER_AI_SIMULATION
                                {
                                    let mut player = PlayerGameData {
                                        model: &thread_model,
                                        name: &model_data.name,
                                        samples: &mut Default::default(),
                                    };

                                    let mut old_player = PlayerGameData {
                                        model: &old_model,
                                        name: "model_old",
                                        samples: &mut Default::default(),
                                    };

                                    last_iter_failed = false;
                                    let mut game = Game::new();
                                    match play_game_to_end(&mut game, &mut player, &mut old_player)
                                    {
                                        Ok(None) => {
                                            drawn
                                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
                    &model_data.name,
                );

                println!(
                    "Model: {}, Vs Initial, Wins (new): {}, Wins (initial): {}, Ties: {}, Time: {:.2}s",
                    model_data.name,
                    white_won,
                    black_won,
                    drawn.load(std::sync::atomic::Ordering::Relaxed),
                    (Instant::now() - st).as_secs_f32(),
                );
            }
        }
    }

    metrics::record_training_status(false);
    provider.shutdown()?;
    Ok(())
}

fn play_game_to_end<'a>(
    g: &mut Game,
    white_player: &'a mut PlayerGameData<'a>,
    black_player: &'a mut PlayerGameData<'a>,
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
        let (player, opponent) = if g.to_play() == Color::White {
            (&mut *white_player, &mut *black_player)
        } else {
            (&mut *black_player, &mut *white_player)
        };

        if g.turn() == hypers::MAX_TURNS_PER_GAME {
            return Err(HiveError::TurnLimitHit);
        }

        let device = player.model.device;

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
            metrics::increment_move_made(Move::Pass, playing, player.name, opponent.name);

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

        // let map = translate_to_valid_moves_mask(&g, &valid_moves, playing, &mut invalid_moves_mask);
        let map = translate_to_valid_moves_mask(&g, &valid_moves, playing, &mut invalid_moves_mask);
        let invalid_moves_tensor = Tensor::from_slice(&invalid_moves_mask);

        let (value, mut policy) = tch::no_grad(|| player.model.value_policy(&curr_state_batch));

        let _ = policy.masked_fill_(&invalid_moves_tensor.to(device), f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");
        g.make_move(*mv)?;

        player.samples.game_state.push(curr_state);
        player.samples.invalid_move_mask.push(invalid_moves_tensor);
        player
            .samples
            .value
            .push(value.view(1i64).to(tch::Device::Cpu));
        player
            .samples
            .selected_policy
            .push(sampled_action_idx.view(1i64).to(tch::Device::Cpu));

        metrics::increment_move_made(*mv, playing, player.name, opponent.name);
    }

    Ok(winner)
}

fn _train_policy_phase(
    adam: &mut Optimizer,
    model: &mut HiveModel,
    frames: &MultipleGames,
    entropy_loss_scaling: f64,
    model_name: &str,
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

    let mut batch_count = 0;
    let mut total_value_loss = Tensor::zeros(1, (Kind::Float, model.device));
    let mut total_value_count = 0;

    model.set_train_mode(true);
    for _ in 0..hypers::POLICY_PHASE_TRAIN_ITERS {
        batch_count += 1;
        let perms = Tensor::randperm(frames.len() as i64, (Kind::Int64, model.device));
        let should_stop_early = false;

        for start in (0..frames.len()).step_by(hypers::BATCH_SIZE) {
            adam.zero_grad();
            let upper = frames.len().min(start + hypers::BATCH_SIZE) as i64;
            let idxs = perms.i((start as i64)..upper);

            let states = state_buffer.index_select(0, &idxs);
            let mask = &mask_buffer.index_select(0, &idxs);

            let (values, mut policies) = model.value_policy(&states);
            let _ = policies.masked_fill_(&mask, f64::NEG_INFINITY);

            let adv = gae_buffer.index_select(0, &idxs);

            // Many of the papers normalize the advantages.
            // In some small scale tests, I'm not seeing a significant advantage in our model.
            // my guess is that it's because the rewards are bounded
            // let adv = (&adv - adv.mean(None)) / (adv.std(false) + 1e-8);

            let logp = policies.log_softmax(1, None).gather(
                1,
                &selections_buffer.index_select(0, &idxs),
                false,
            );

            let logp_old = logp_old.index_select(0, &idxs);

            let ratio = (&logp - &logp_old).exp_();
            let ratio_f = f64::try_from(ratio.mean(None)).unwrap();
            let adv_f = f64::try_from(adv.mean(None)).unwrap();
            let clip_adv = ratio.clamp(1.0 - hypers::EPSILON, 1.0 + hypers::EPSILON) * &adv;

            let pi_loss = (ratio * adv).minimum(&clip_adv).mean(None);

            let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                .square()
                .mean(None);

            total_value_loss = tch::no_grad(|| total_value_loss + &value_loss);
            total_value_count += 1;

            let p_prob = policies.softmax(1, None) + 0.00001;
            let h = mask.logical_not() * (&p_prob * p_prob.log());
            let entropy_loss = h.sum_dim_intlist(1, false, None).neg().mean(None);
            let approx_kl: f64 = (logp_old - logp).mean(None).try_into().expect("success");
            // Tensor::stack(&[&value_loss, &pi_loss, &entropy_loss], 0).print();

            metrics::record_policy_minibatch_statistics(
                f64::try_from(&value_loss).unwrap(),
                f64::try_from(&pi_loss).unwrap(),
                f64::try_from(&entropy_loss).unwrap(),
                ratio_f,
                adv_f,
                approx_kl,
                model_name,
            );

            let loss: Tensor = value_loss
                - (hypers::PI_LOSS_RATIO * pi_loss)
                - (entropy_loss_scaling * entropy_loss);

            loss.backward();
            adam.clip_grad_norm(hypers::GRAD_CLIP);
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

fn _train_auxiliary_phase(
    adam: &mut Optimizer,
    model: &mut HiveModel,
    frames: &MultipleGames,
    model_name: &str,
) -> usize {
    let state_buffer = Tensor::stack(frames.game_state.as_slice(), 0).to(model.device);
    let mask_buffer = Tensor::stack(frames.invalid_move_mask.as_slice(), 0).to(model.device);
    let target_value_buffer = Tensor::stack(frames.target_value.as_slice(), 0).to(model.device);
    const LARGE_NEGATIVE: f64 = -1.0e9;

    let pi_old = tch::no_grad(|| model.policy(&state_buffer))
        .masked_fill_(&mask_buffer, LARGE_NEGATIVE)
        .log_softmax(1, None);

    let mut batch_count = 0;
    // let mut total_value_loss = Tensor::zeros(1, (Kind::Float, model.device));
    // let mut total_value_count = 0;

    model.set_train_mode(true);
    for _ in 0..hypers::AUXILIARY_PHASE_TRAIN_ITERS {
        batch_count += 1;
        let perms = Tensor::randperm(frames.len() as i64, (Kind::Int64, model.device));

        for start in (0..frames.len()).step_by(hypers::BATCH_SIZE) {
            adam.zero_grad();
            let upper = frames.len().min(start + hypers::BATCH_SIZE) as i64;
            let idxs = perms.i((start as i64)..upper);

            let states = state_buffer.index_select(0, &idxs);
            let mask = &mask_buffer.index_select(0, &idxs);

            let (values, mut policies) = model.value_policy_auxiliary(&states);
            let pi = policies
                .masked_fill_(&mask, LARGE_NEGATIVE)
                .log_softmax(1, None);

            // Many of the papers normalize the advantages.
            // In some small scale tests, I'm not seeing a significant advantage in our model.
            // my guess is that it's because the rewards are bounded
            // let adv = (&adv - adv.mean(None)) / (adv.std(false) + 1e-8);

            let pi_old = pi_old.index_select(0, &idxs);
            let kl = pi.kl_div(&pi_old, tch::Reduction::None, true);
            let kl_loss = kl.sum_dim_intlist(1, false, None).mean(None);

            let value_loss = (values - target_value_buffer.index_select(0, &idxs))
                .square()
                .mean(None);

            metrics::record_auxiliary_minibatch_statistics(
                f64::try_from(&value_loss).unwrap(),
                0.0,
                f64::try_from(&kl_loss).unwrap(),
                model_name,
            );

            let loss: Tensor =
                hypers::AUXILIARY_VALUE_SCALE * value_loss + hypers::AUXILIARY_BETA_CLONE * kl_loss;

            loss.backward();
            adam.step();
        }
    }

    model.set_train_mode(false);

    // let mse = total_value_loss / total_value_count;
    // mse.print();
    // let mse: f64 = mse.try_into().unwrap();

    // metrics::record_value_mse(mse);

    // metrics::record_training_batches(batch_count);
    batch_count as usize
}
