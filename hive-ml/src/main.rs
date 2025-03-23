use std::collections::HashMap;
use std::time::Instant;

use hive_engine::game::GameWinner;
use hive_engine::movement::Move;
use hive_engine::BOARD_SIZE;
use hive_engine::{game::Game, piece::Color};
use hive_ml::hypers::{self, INPUT_ENCODED_DIMS, OUTPUT_LENGTH};
use hive_ml::{translate_to_tensor, PieceEncodable};
use tch::kind::FLOAT_CPU;
use tch::nn::{OptimizerConfig, VarStore};
use tch::{nn, Kind, Tensor};

// input data shape should be [batch, channels, rows, cols]
// output data shape should be [batch, ]

pub struct HiveModel {
    shared_layers: tch::nn::Sequential,
    policy_layer: tch::nn::Sequential,
    value_layer: tch::nn::Sequential,
    device: tch::Device,
}
struct GameFrame {
    playing: Color,
    game_state: Tensor,
    /// A 1x1 tensor that contains the index sampled by the policy.
    selected_policy: Tensor,
    /// A mask that represents the actions that are invalid in the current state.
    invalid_move_mask: Tensor,
    /// A tensor that represents the rewards we got from making this decision.
    /// Not time discounted.
    future_value: Tensor,
    time_adjusted_value: Tensor,
}

impl HiveModel {
    pub fn new(p: &nn::Path) -> Self {
        let stride = |s| nn::ConvConfig {
            stride: s,
            ..Default::default()
        };

        let shared_layers = nn::seq()
            .add(nn::conv2d(
                p / "c1",
                INPUT_ENCODED_DIMS as i64,
                32,
                3,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, stride(2)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 2304, 512, Default::default()))
            .add_fn(|xs| xs.relu());

        let value_layer = nn::seq()
            .add(nn::linear(p / "c1", 512, 1, Default::default()))
            .add_fn(|xs| 2 * xs.sigmoid() - 1.0);

        let policy_layer = nn::seq().add(nn::linear(
            p / "al",
            512,
            OUTPUT_LENGTH as i64,
            Default::default(),
        ));

        let device = p.device();

        Self {
            shared_layers,
            policy_layer,
            value_layer,
            device,
        }
    }

    pub fn value_policy(&self, game_state: &Tensor) -> (Tensor, Tensor) {
        let t = self.shared_layers(game_state);
        (t.apply(&self.value_layer), t.apply(&self.policy_layer))
    }

    pub fn value(&self, game_state: &Tensor) -> Tensor {
        let t = self.shared_layers(game_state);
        t.apply(&self.value_layer)
    }

    pub fn policy(&self, game_state: &Tensor) -> Tensor {
        let t = self.shared_layers(game_state);
        t.apply(&self.policy_layer)
    }

    fn shared_layers(&self, game_state: &Tensor) -> Tensor {
        game_state.apply(&self.shared_layers)
    }
}

pub fn main() -> hive_engine::Result<()> {
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let mut model = HiveModel::new(&vs.root());

    let mut buffer = Vec::new();
    for _ in 0..100 {
        let mut g = Game::new();

        let st = Instant::now();
        play_game_to_end(&mut g, &model, &mut buffer)?;
        println!(
            "Game Over, Turns: {}, Time Taken: {}s",
            g.turn(),
            (Instant::now() - st).as_secs_f32()
        );

        // Do a training loop.
        let st = Instant::now();
        _train_loop(&vs, &mut model, buffer.as_slice());
        println!(
            "Training Iteration Complete, Time Taken: {}s",
            (Instant::now() - st).as_secs_f32()
        );

        buffer.clear();
    }

    Ok(())
}

fn play_game_to_end(
    g: &mut Game,
    model: &HiveModel,
    samples: &mut Vec<GameFrame>,
) -> hive_engine::Result<()> {
    let mut consecutive_passes = 0;

    let mut valid_moves = Vec::new();

    let mut invalid_moves_mask = vec![true; OUTPUT_LENGTH];

    let mut winner = None;
    loop {
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
        let curr_state = translate_to_tensor(&g, playing);
        let curr_state_batch = curr_state.view((
            1,
            INPUT_ENCODED_DIMS as i64,
            BOARD_SIZE as i64,
            BOARD_SIZE as i64,
        ));
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

        let (value, mut policy) = tch::no_grad(|| model.value_policy(&curr_state_batch));

        let _ = policy.masked_fill_(&invalid_moves_tensor, f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");
        g.make_move(**mv)?;

        samples.push(GameFrame {
            playing,
            game_state: curr_state,
            invalid_move_mask: invalid_moves_tensor,
            future_value: value.view(1i64),
            time_adjusted_value: value.view(1i64),
            selected_policy: sampled_action_idx.view(()),
        });
    }

    const MAX_TD_DISTANCE: usize = 10;
    let gamma = Tensor::from(0.99f32);
    let max_decay_gamma = &gamma.pow_tensor_scalar(MAX_TD_DISTANCE as i64);

    let final_score = match winner {
        Some(..) => &Tensor::from(0.5f32),
        None => &Tensor::from(0.0f32),
    };

    let num_samples = samples.len();
    for idx in 0..num_samples {
        if idx + MAX_TD_DISTANCE < num_samples {
            // Use the value estimate as the target.
            let future_frame = &samples[idx + MAX_TD_DISTANCE];
            let future_color = future_frame.playing;
            let mut future_score = future_frame.future_value.copy();

            let sample = &mut samples[idx];
            sample.future_value = if future_color != sample.playing {
                future_score.neg_()
            } else {
                future_score
            };

            sample.time_adjusted_value = sample.future_value.multiply(max_decay_gamma);
        } else {
            // use the true value as the target.
            let decay_factor = (num_samples - idx - 1) as i64;
            let decay_factor = gamma.pow_tensor_scalar(decay_factor);
            let mut future_score = final_score.copy();

            let sample = &mut samples[idx];

            sample.future_value = if winner != Some(sample.playing) {
                // If the result was a draw, then we'll just have -0.0 which is still fine.
                future_score.neg_()
            } else {
                future_score
            };

            sample.time_adjusted_value = sample.future_value.multiply(&decay_factor);
        }
    }

    Ok(())
}

fn _train_loop(vs: &VarStore, model: &mut HiveModel, frames: &[GameFrame]) {
    let mut adam = nn::Adam::default().build(vs, 1e-4).unwrap();

    let options = (tch::Kind::Float, model.device);
    let mut state_buffer = {
        let mut state_buffer_shape = vec![frames.len() as i64];
        state_buffer_shape.extend_from_slice(frames[0].game_state.size().as_slice());
        Tensor::zeros(state_buffer_shape, options)
    };

    let mut mask_buffer = Tensor::zeros(
        [frames.len() as i64, hypers::OUTPUT_LENGTH as i64],
        (tch::Kind::Bool, model.device),
    );
    let mut selections_buffer = Tensor::zeros(
        [frames.len() as i64, 1 as i64],
        (tch::Kind::Int64, model.device),
    );

    // The target value is used by the value loss to compute the mse between the value function and the
    // correct value.
    let mut target_values_buffer = Tensor::zeros([frames.len() as i64, 1], options);

    // The future value is used by the policy loss to compute the advantage adv = value_fut - value_pres
    let mut future_values_buffer = Tensor::zeros([frames.len() as i64, 1], options);

    for (idx, frame) in frames.iter().enumerate() {
        let idx = idx as i64;
        let _ = state_buffer.index_put_(&[Some(Tensor::from(idx))], &frame.game_state, false);
        let _ = mask_buffer.index_put_(&[Some(Tensor::from(idx))], &frame.invalid_move_mask, false);
        let _ = target_values_buffer.index_put_(
            &[Some(Tensor::from(idx))],
            &frame.time_adjusted_value,
            false,
        );

        let _ =
            future_values_buffer.index_put_(&[Some(Tensor::from(idx))], &frame.future_value, false);

        let _ =
            selections_buffer.index_put_(&[Some(Tensor::from(idx))], &frame.selected_policy, false);
    }

    let logp_old = tch::no_grad(|| model.policy(&state_buffer))
        .masked_fill_(&mask_buffer, f64::NEG_INFINITY)
        .log_softmax(1, None)
        .gather(1, &selections_buffer, false);

    let clip_ratio = 0.1f64;
    let value_policy_learning_ratio = 0.1f64;
    for _ in 0..6 {
        adam.zero_grad();
        let idxs = Tensor::randint(frames.len() as i64, &[64i64], (Kind::Int64, model.device));
        let states = state_buffer.index_select(0, &idxs);
        let (values, mut policies) = model.value_policy(&states);

        let adv = future_values_buffer.index_select(0, &idxs) - &values;

        let logp = policies
            .masked_fill_(&mask_buffer.index_select(0, &idxs), f64::NEG_INFINITY)
            .log_softmax(1, None)
            .gather(1, &selections_buffer.index_select(0, &idxs), false);

        let logp_old = logp_old.index_select(0, &idxs);

        let ratio = (logp - logp_old).exp_();
        let clip_adv = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * &adv;
        let pi_loss = -(ratio * adv).minimum(&clip_adv).mean(None);

        let value_loss = (values - target_values_buffer.index_select(0, &idxs))
            .square()
            .mean(None);

        let loss = value_policy_learning_ratio * value_loss + pi_loss;

        loss.backward();
        adam.step();
    }

    adam.zero_grad();
}
