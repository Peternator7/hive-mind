use std::sync::Mutex;

use encode::{translate_game_to_conv_tensor, translate_to_valid_moves_mask};
use frames::SingleGame;
use hive_engine::{game::Game, movement::Move, piece::Color};
use model::{HiveModel, Prediction};
use rand::{rngs::ThreadRng, seq::IndexedRandom, Rng};
use seen::SeenPositions;

pub mod acc;
pub mod encode;
pub mod frames;
pub mod hypers;
pub mod metrics;
pub mod model;
pub mod model2;
pub mod seen;

pub trait Agent {
    fn name(&self) -> &str;

    fn set_color(&mut self, color: Color);

    fn select_move(&mut self, g: &Game) -> hive_engine::Result<Move>;

    fn observe_final_state(&mut self, g: &Game) {
        let _ = g;
    }
}

impl<T> Agent for &mut T
where
    T: Agent,
{
    fn name(&self) -> &str {
        <T as Agent>::name(self)
    }

    fn set_color(&mut self, color: Color) {
        <T as Agent>::set_color(self, color)
    }

    fn select_move(&mut self, g: &Game) -> hive_engine::Result<Move> {
        <T as Agent>::select_move(self, g)
    }
}

pub struct RandomAgent {
    playing_as: Color,
    move_buf: Vec<Move>,
    rng: rand::prelude::ThreadRng,
}

impl RandomAgent {
    pub fn new() -> Self {
        Self {
            playing_as: Color::Black,
            move_buf: Default::default(),
            rng: rand::rng(),
        }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "model_random"
    }

    fn set_color(&mut self, color: Color) {
        self.playing_as = color;
    }

    fn select_move(&mut self, g: &Game) -> hive_engine::Result<Move> {
        self.move_buf.clear();
        g.load_all_potential_moves(&mut self.move_buf)?;
        let output = self
            .move_buf
            .choose(&mut self.rng)
            .copied()
            .unwrap_or(Move::Pass);

        self.move_buf.clear();
        Ok(output)
    }
}

pub struct HiveModelAgent<'a> {
    playing_as: Color,
    is_first_move: bool,
    capture_moves: bool,
    valid_moves: Vec<Move>,
    invalid_moves_mask: Vec<bool>,
    name: &'a str,
    model: &'a HiveModel,
    samples: &'a mut SingleGame,
    seen: Option<&'a Mutex<SeenPositions>>,
}

impl<'a> HiveModelAgent<'a> {
    pub fn new(
        name: &'a str,
        model: &'a HiveModel,
        samples: &'a mut SingleGame,
        seen: Option<&'a Mutex<SeenPositions>>,
        capture_moves: bool,
    ) -> Self {
        Self {
            playing_as: Color::Black,
            is_first_move: true,
            capture_moves,
            samples,
            seen,
            model,
            name,
            invalid_moves_mask: vec![true; hypers::OUTPUT_LENGTH],
            valid_moves: Vec::new(),
        }
    }
}

impl<'a> Agent for HiveModelAgent<'a> {
    fn name(&self) -> &str {
        self.name
    }

    fn set_color(&mut self, color: Color) {
        self.playing_as = color;
    }

    fn select_move(&mut self, g: &Game) -> hive_engine::Result<Move> {
        let device = self.model.device;
        self.valid_moves.clear();
        self.invalid_moves_mask.fill(true);

        let playing = g.to_play();

        g.load_all_potential_moves(&mut self.valid_moves)?;
        if self.valid_moves.is_empty() {
            return Ok(Move::Pass);
        }

        let curr_state = translate_game_to_conv_tensor(&g, playing);
        let curr_state_batch = curr_state.unsqueeze(0).to(device);

        // let map = translate_to_valid_moves_mask(&g, &valid_moves, playing, &mut invalid_moves_mask);
        let map = translate_to_valid_moves_mask(
            &g,
            &self.valid_moves,
            playing,
            &mut self.invalid_moves_mask,
        );
        let invalid_moves_tensor = tch::Tensor::from_slice(&self.invalid_moves_mask);

        let Prediction {
            value,
            mut policy,
            intrinsic_value,
        } = tch::no_grad(|| self.model.predict(&curr_state_batch));

        let novelty = tch::no_grad(|| self.model.novelty(&curr_state_batch));

        let _ = policy.masked_fill_(&invalid_moves_tensor.to(device), f64::NEG_INFINITY);

        let sampled_action_idx = policy.softmax(-1, None).multinomial(1, true);
        let action_prob: i64 = i64::try_from(&sampled_action_idx).expect("cast");
        let mv = map.get(&(action_prob as usize)).expect("populated above.");

        if self.capture_moves {
            self.samples.game_state.push(curr_state);
            self.samples.invalid_move_mask.push(invalid_moves_tensor);
            self.samples
                .value
                .push(value.view(1i64).to(tch::Device::Cpu));

            self.samples
                .selected_policy
                .push(sampled_action_idx.view(1i64).to(tch::Device::Cpu));

            self.samples
                .intrinsic_value
                .push(intrinsic_value.view(1i64).to(tch::Device::Cpu));

            // if (g.turn() > 1 && playing == Color::White) || (g.turn() > 0 && playing == Color::Black) {
            if self.is_first_move {
                self.is_first_move = false;
            } else {
                self.samples
                    .novelty
                    .push(novelty.view(1i64).to(tch::Device::Cpu));
            }
        }

        if let Some(ref seen) = self.seen {
            if seen.lock().unwrap().is_unseen_position(g) {
                metrics::increment_novel_position(self.playing_as, self.name);
            } else {
                metrics::increment_seen_position(self.playing_as, self.name);
            }
        }

        Ok(*mv)
    }

    fn observe_final_state(&mut self, g: &Game) {
        if self.capture_moves {
            let gw = translate_game_to_conv_tensor(&g, self.playing_as);
            let gw_batch = gw.unsqueeze(0).to(self.model.device);
            let final_novelty = tch::no_grad(|| self.model.novelty(&gw_batch));

            self.samples
                .novelty
                .push(final_novelty.view(1).to(tch::Device::Cpu));

            assert!(self.samples.validate_buffers());
        }
    }
}

pub struct BlendedAgent<A, B> {
    name: String,
    model_a: A,
    model_b: B,
    model_b_prob: f64,
    rng: ThreadRng,
}

impl<A, B> BlendedAgent<A, B>
where
    A: Agent,
    B: Agent,
{
    pub fn new(name: &str, model_a: A, model_b: B, model_b_prob: f64) -> Self {
        assert!(model_b_prob >= 0.0);
        assert!(model_b_prob <= 1.0);

        Self {
            name: name.to_string(),
            model_a,
            model_b,
            model_b_prob,
            rng: rand::rng(),
        }
    }
}

impl<A, B> Agent for BlendedAgent<A, B>
where
    A: Agent,
    B: Agent,
{
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn set_color(&mut self, color: Color) {
        self.model_a.set_color(color);
        self.model_b.set_color(color);
    }

    fn select_move(&mut self, g: &Game) -> hive_engine::Result<Move> {
        if self.rng.random_bool(self.model_b_prob) {
            self.model_b.select_move(g)
        } else {
            self.model_a.select_move(g)
        }
    }

    fn observe_final_state(&mut self, g: &Game) {
        self.model_a.observe_final_state(g);
        self.model_b.observe_final_state(g);
    }
}
