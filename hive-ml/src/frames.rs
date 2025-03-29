use hive_engine::piece::Color;
use tch::Tensor;

#[derive(Debug, Default)]
pub struct MultipleGames {
    pub game_state: Vec<Tensor>,

    pub sequence_length: Vec<Tensor>,

    /// A 1x1 tensor that contains the index sampled by the policy.
    pub selected_policy: Vec<Tensor>,

    /// A mask that represents the actions that are invalid in the current state.
    pub invalid_move_mask: Vec<Tensor>,

    /// A tensor that represents the rewards we got from making this decision.
    /// Not time discounted.
    pub gae: Vec<Tensor>,

    pub target_value: Vec<Tensor>,
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
            && len == self.sequence_length.len()
    }

    pub fn ingest_game(
        &mut self,
        other: &mut SingleGame,
        winner: Option<Color>,
        gamma: f64,
        lambda: f64,
        max_frames_per_game: usize,
    ) {
        assert!(other.validate_buffers());

        let gl = gamma * lambda;
        let mut value = std::mem::take(&mut other.value);

        // Flip the signs on the value estimator so that everything is from white's perspective.
        for idx in 0..value.len() {
            if other.playing[idx] == Color::Black {
                let _ = value[idx].neg_();
            }
        }

        let mut gae_values = Vec::with_capacity(value.len());

        let mut gae = match winner {
            None => Tensor::from(0.0f32),
            Some(Color::White) => Tensor::from(0.5f32),
            Some(Color::Black) => Tensor::from(-0.5f32),
        };

        let mut discounted_rewards = gae.copy();

        for idx in (0..value.len()).rev() {
            if idx == value.len() - 1 {
                gae = gae - &value[idx];
                gae_values.push(gae.copy());
                continue;
            }

            let delta = gamma * &value[idx + 1] - &value[idx];
            value[idx + 1] = discounted_rewards.copy();
            discounted_rewards = discounted_rewards * gamma;

            gae = delta + gl * &gae;
            gae_values.push(gae.copy());
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
        if value.len() > max_frames_per_game {
            samples_to_skip = value.len() - max_frames_per_game;
        }

        self.game_state.extend(other.game_state.drain(..).skip(samples_to_skip));
        self.sequence_length.extend(other.seq_length.drain(..).skip(samples_to_skip));
        self.selected_policy.extend(other.selected_policy.drain(..).skip(samples_to_skip));
        self.invalid_move_mask
            .extend(other.invalid_move_mask.drain(..).skip(samples_to_skip));
        self.target_value.extend(value.into_iter().skip(samples_to_skip));
        self.gae.extend(gae_values.into_iter().skip(samples_to_skip));

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

    pub seq_length: Vec<Tensor>,
}

impl SingleGame {
    pub fn clear(&mut self) {
        self.playing.clear();
        self.game_state.clear();
        self.selected_policy.clear();
        self.invalid_move_mask.clear();
        self.value.clear();
        self.seq_length.clear();
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
            && len == self.seq_length.len()
    }
}
