use hive_engine::piece::Color;
use rand::{rngs::StdRng, SeedableRng};
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

    /// A reusable buffer so that we can sample frames from played games more randomly.
    _sample_buffer: Vec<usize>,
    _rng: StdRng,
}

impl Default for MultipleGames {
    fn default() -> Self {
        Self {
            game_state: Default::default(),
            selected_policy: Default::default(),
            invalid_move_mask: Default::default(),
            gae: Default::default(),
            target_value: Default::default(),
            _sample_buffer: Default::default(),
            _rng: StdRng::from_seed([0; 32]),
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

    pub fn ingest_game(
        &mut self,
        other: &mut SingleGame,
        winner: Option<Color>,
        mut gamma: f64,
        lambda: f64,
        max_frames_per_game: usize,
    ) {
        assert!(other.validate_buffers());

        // If we have a short game, we should re-scale the rewards so that the
        // value at turn 0 is about 0.
        if other.len() < hypers::APPROXIMATE_TURN_MEMORY {
            gamma = 1.0 - (1.0 / other.len() as f64);
        }

        let gl = gamma * lambda;
        let mut value = std::mem::take(&mut other.value);

        // Flip the signs on the value estimator so that everything is from white's perspective.
        for idx in 0..value.len() {
            if other.playing[idx] == Color::Black {
                let _ = value[idx].neg_();
            }
        }

        let mut gae_values = Vec::with_capacity(value.len());
        // let mut distance_to_end = 0;

        let mut gae = match winner {
            None => Tensor::from(0.0f32),
            Some(Color::White) => Tensor::from(1.0f32),
            Some(Color::Black) => Tensor::from(-1.0f32),
        };

        let mut discounted_rewards = gae.copy();

        for idx in (0..value.len()).rev() {
            if idx == value.len() - 1 {
                gae = gae - &value[idx];
                gae_values.push(gae.copy());
                continue;
            }

            // let mut long_game_penalty: Tensor = {
            //     let x = distance_to_end - hypers::PENALIZE_TURNS_DISTANCE_FROM_END;
            //     let x = x as f64;
            //     0.333 * Tensor::from(x / 2.0).sigmoid()
            // };

            // if idx < hypers::PENALIZE_TURNS_DISTANCE_FROM_END as usize {
            //     let scale = idx as f64  / hypers::PENALIZE_TURNS_DISTANCE_FROM_END as f64;
            //     long_game_penalty *= scale * scale;
            // };

            let delta = gamma * &value[idx + 1] - &value[idx];
            value[idx + 1] = discounted_rewards.copy();
            discounted_rewards = discounted_rewards * gamma;

            gae = delta + gl * &gae;
            // We apply an additional penalty for moves in very long running games.
            // because there's a reasonable argument that they didn't accomplish very much.
            gae_values.push(gae.clamp(-1.0, 1.0));

            // distance_to_end += 1;
        }

        value[0] = discounted_rewards;

        gae_values.reverse();
        for idx in 0..gae_values.len() {
            if other.playing[idx] == Color::Black {
                let _ = gae_values[idx].neg_();
                let _ = value[idx].neg_();
            }
        }

        let mut samples_to_skip = 0;
        if value.len() > 2 * max_frames_per_game {
            samples_to_skip = value.len() - 2 * max_frames_per_game;
        }

        self.game_state
            .extend(other.game_state.drain(..).skip(samples_to_skip));
        self.selected_policy
            .extend(other.selected_policy.drain(..).skip(samples_to_skip));
        self.invalid_move_mask
            .extend(other.invalid_move_mask.drain(..).skip(samples_to_skip));
        self.target_value
            .extend(value.into_iter().skip(samples_to_skip));
        self.gae
            .extend(gae_values.into_iter().skip(samples_to_skip));

        other.playing.clear();

        assert!(self.validate_buffers());
    }
}

#[derive(Debug, Default)]
pub struct SingleGame {
    pub playing: Vec<Color>,
    pub game_state: Vec<Tensor>,

    /// A 1x1 tensor that contains the index sampled by the policy.
    pub selected_policy: Vec<Tensor>,

    /// A mask that represents the actions that are invalid in the current state.
    pub invalid_move_mask: Vec<Tensor>,

    pub value: Vec<Tensor>,
}

impl SingleGame {
    pub fn clear(&mut self) {
        self.playing.clear();
        self.game_state.clear();
        self.selected_policy.clear();
        self.invalid_move_mask.clear();
        self.value.clear();
    }

    pub fn len(&self) -> usize {
        self.playing.len()
    }

    pub fn validate_buffers(&self) -> bool {
        let len = self.playing.len();
        len == self.game_state.len()
            && len == self.selected_policy.len()
            && len == self.invalid_move_mask.len()
            && len == self.value.len()
    }
}
