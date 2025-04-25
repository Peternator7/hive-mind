use hive_engine::piece::Color;
use rand::{rngs::SmallRng, SeedableRng};
use tch::Tensor;

use crate::hypers;

#[derive(Debug)]
pub struct MultipleGames {
    pub game_state: Vec<Tensor>,

    /// A 1x1 tensor that contains the index sampled by the policy.
    pub selected_policy: Vec<Tensor>,

    /// A mask that represents the actions that are invalid in the current state.
    pub invalid_move_mask: Vec<Tensor>,
    /// A tensor that represents the rewards we got from making this decision.
    /// Not time discounted.
    pub gae: Vec<Tensor>,
    pub target_value: Vec<Tensor>,
    rng: SmallRng,
}

struct Frame {
    td_error: f64,
    game_state: Tensor,
    selected_policy: Tensor,
    invalid_move_mask: Tensor,
    gae: Tensor,
    target_value: Tensor,
}

impl Default for MultipleGames {
    fn default() -> Self {
        Self {
            game_state: Default::default(),
            selected_policy: Default::default(),
            invalid_move_mask: Default::default(),
            gae: Default::default(),
            target_value: Default::default(),
            rng: rand::rngs::SmallRng::from_seed(Default::default()),
        }
    }
}

impl MultipleGames {
    pub fn clear(&mut self) {
        self.game_state.clear();
        self.selected_policy.clear();
        self.invalid_move_mask.clear();
        self.gae.clear();
        self.target_value.clear();
    }

    pub fn len(&self) -> usize {
        self.game_state.len()
    }

    pub fn validate_buffers(&self) -> bool {
        let len = self.game_state.len();
        len == self.selected_policy.len()
            && len == self.invalid_move_mask.len()
            && len == self.gae.len()
            && len == self.target_value.len()
    }

    pub fn ingest_multiple_games(&mut self, other: &mut MultipleGames, step_by: usize) {
        self.game_state
            .extend(other.game_state.drain(..).step_by(step_by));
        self.selected_policy
            .extend(other.selected_policy.drain(..).step_by(step_by));
        self.invalid_move_mask
            .extend(other.invalid_move_mask.drain(..).step_by(step_by));
        self.gae.extend(other.gae.drain(..).step_by(step_by));
        self.target_value
            .extend(other.target_value.drain(..).step_by(step_by));
    }

    pub fn ingest_game(
        &mut self,
        other: &mut SingleGame,
        winner: Option<Color>,
        stalled: bool,
        mut gamma: f64,
        lambda: f64,
        max_frames_per_game: usize,
    ) {
        assert!(other.validate_buffers());
        let playing = other.playing.unwrap();

        // If we have a short game, we should re-scale the rewards so that the
        // value at turn 0 is about 0.
        if other.len() < hypers::APPROXIMATE_TURN_MEMORY {
            gamma = 1.0 - (1.0 / other.len() as f64);
        }

        let gl = gamma * lambda;
        let mut value = std::mem::take(&mut other.value);

        let mut gae_values = Vec::with_capacity(value.len());
        let mut td_error = Vec::with_capacity(value.len());

        // This is the rewards for the final step.
        let mut gae = match winner {
            None if stalled => Tensor::from(hypers::PENALTY_FOR_TIMING_OUT),
            None => Tensor::from(0.0f32),
            Some(winner_color) if winner_color == playing => Tensor::from(1.0f32),
            Some(..) => Tensor::from(-1.0f32),
        };

        let mut discounted_rewards = gae.copy();

        for idx in (0..value.len()).rev() {
            if idx == value.len() - 1 {
                gae = gae - &value[idx];
                gae_values.push(gae.copy());
                td_error.push(f64::try_from((&discounted_rewards - &value[idx]).square()).unwrap());
                continue;
            }

            let curr_step_rewards = -hypers::PENALTY_FOR_MOVING;
            let delta = gamma * &value[idx + 1] + curr_step_rewards - &value[idx];
            value[idx + 1] = discounted_rewards.copy();
            discounted_rewards = curr_step_rewards + discounted_rewards * gamma;

            gae = (delta + gl * &gae).clamp(-1.0, 1.0);
            gae_values.push(gae.copy());
            td_error.push(f64::try_from((&discounted_rewards - &value[idx]).square()).unwrap());
        }

        value[0] = discounted_rewards;

        gae_values.reverse();
        td_error.reverse();

        let game_state = other.game_state.drain(..);
        let mut selected_policy = other.selected_policy.drain(..);
        let mut invalid_move_mask = other.invalid_move_mask.drain(..);
        let mut value = value.into_iter();
        let mut gae_values = gae_values.into_iter();
        let mut td_error = td_error.into_iter();

        let mut row_major_frames = Vec::new();
        for game_state in game_state {
            row_major_frames.push(Some(Frame {
                game_state,
                selected_policy: selected_policy.next().unwrap(),
                invalid_move_mask: invalid_move_mask.next().unwrap(),
                target_value: value.next().unwrap(),
                gae: gae_values.next().unwrap(),
                td_error: td_error.next().unwrap(),
            }))
        }

        let indices = rand::seq::index::sample_weighted(
            &mut self.rng,
            row_major_frames.len(),
            |i| row_major_frames[i].as_ref().unwrap().td_error,
            row_major_frames.len().min(max_frames_per_game),
        )
        .unwrap();
        for idx in indices {
            let frame = row_major_frames[idx].take().unwrap();
            self.game_state.push(frame.game_state);
            self.selected_policy.push(frame.selected_policy);
            self.invalid_move_mask.push(frame.invalid_move_mask);
            self.gae.push(frame.gae);
            self.target_value.push(frame.target_value);
        }

        // let mut samples_to_skip = 0;
        // if value.len() > 2 * max_frames_per_game {
        //     samples_to_skip = value.len() - 2 * max_frames_per_game;
        // }
        // self.game_state
        //     .extend(other.game_state.drain(..).skip(samples_to_skip).step_by(2));
        // self.selected_policy.extend(
        //     other
        //         .selected_policy
        //         .drain(..)
        //         .skip(samples_to_skip)
        //         .step_by(2),
        // );
        // self.invalid_move_mask.extend(
        //     other
        //         .invalid_move_mask
        //         .drain(..)
        //         .skip(samples_to_skip)
        //         .step_by(2),
        // );
        // self.target_value
        //     .extend(value.into_iter().skip(samples_to_skip).step_by(2));
        // self.gae
        //     .extend(gae_values.into_iter().skip(samples_to_skip).step_by(2));

        other.playing.take();

        assert!(self.validate_buffers());
    }
}

#[derive(Debug, Default)]
pub struct SingleGame {
    pub playing: Option<Color>,
    pub game_state: Vec<Tensor>,

    /// A 1x1 tensor that contains the index sampled by the policy.
    pub selected_policy: Vec<Tensor>,

    /// A mask that represents the actions that are invalid in the current state.
    pub invalid_move_mask: Vec<Tensor>,

    pub value: Vec<Tensor>,
}

impl SingleGame {
    pub fn clear(&mut self) {
        self.playing.take();
        self.game_state.clear();
        self.selected_policy.clear();
        self.invalid_move_mask.clear();
        self.value.clear();
    }

    pub fn len(&self) -> usize {
        self.game_state.len()
    }

    pub fn validate_buffers(&self) -> bool {
        let len = self.game_state.len();
        len == self.selected_policy.len()
            && len == self.invalid_move_mask.len()
            && len == self.value.len()
    }
}
