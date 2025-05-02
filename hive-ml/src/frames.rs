use std::sync::atomic::AtomicUsize;

use hive_engine::piece::Color;
use rand::{rngs::SmallRng, SeedableRng};
use tch::Tensor;

use crate::hypers;

static GAMECOUNT: AtomicUsize = AtomicUsize::new(0);

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
    pub gae_intrinsic: Vec<Tensor>,

    pub target_value: Vec<Tensor>,
    pub intrinsic_value: Vec<Tensor>,
    rng: SmallRng,
}

struct Frame {
    td_error: f64,
    game_state: Tensor,
    selected_policy: Tensor,
    invalid_move_mask: Tensor,
    gae: Tensor,
    gae_intrinsic: Tensor,
    target_value: Tensor,
    intrinsic_value: Tensor,
}

impl Default for MultipleGames {
    fn default() -> Self {
        Self {
            game_state: Default::default(),
            selected_policy: Default::default(),
            invalid_move_mask: Default::default(),
            gae: Default::default(),
            gae_intrinsic: Default::default(),
            target_value: Default::default(),
            intrinsic_value: Default::default(),
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
        self.gae_intrinsic.clear();
        self.intrinsic_value.clear();
        self.target_value.clear();
    }

    pub fn len(&self) -> usize {
        self.game_state.len()
    }

    pub fn validate_buffers(&self) -> bool {
        let len = self.game_state.len();
        assert_eq!(len, self.selected_policy.len());
        assert_eq!(len, self.invalid_move_mask.len());
        assert_eq!(len, self.gae.len());
        assert_eq!(len, self.target_value.len());
        assert_eq!(len, self.gae_intrinsic.len());
        true
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

        other.clear();
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

        let intrinsic_gamma = 0.95 * gamma;
        let intrinsic_gl = lambda * intrinsic_gamma;

        // The novelty frames are one off from the rest of the values because they lag
        // by a turn.

        let gl = gamma * lambda;
        let mut values_external = std::mem::take(&mut other.value);
        let mut values_intrinsic = std::mem::take(&mut other.intrinsic_value);

        let mut adv_intrinsic = Vec::with_capacity(values_external.len());
        let mut adv_external = Vec::with_capacity(values_external.len());
        let mut td_error = Vec::with_capacity(values_external.len());

        // This is the rewards for the final step.
        let outcome_rewards = match winner {
            None if stalled => Tensor::from(hypers::PENALTY_FOR_TIMING_OUT),
            None => Tensor::from(0.0f32),
            Some(winner_color) if winner_color == playing => Tensor::from(1.0f32),
            Some(..) => Tensor::from(-1.0f32),
        };

        let mut gae_external = outcome_rewards.copy();
        let mut gae_intrinsic = Tensor::from(0.0);

        let mut td_value_external = outcome_rewards.copy();
        let mut td_value_intrinsic = Tensor::from(0.0).unsqueeze(0);

        for idx in (0..values_external.len()).rev() {
            if idx == values_external.len() - 1 {
                gae_external = gae_external - &values_external[idx];
                gae_intrinsic = other.novelty.last().unwrap() - values_intrinsic[idx].copy();
                adv_external.push(gae_external.copy());
                adv_intrinsic.push(gae_intrinsic.copy());

                td_error.push(
                    f64::try_from((&td_value_external - &values_external[idx]).square()).unwrap(),
                );

                continue;
            }

            // Calculate the intrinsic rewards.
            let rewards_intrinsic = other.novelty[idx].copy() - hypers::PENALTY_FOR_MOVING;
            let delta = intrinsic_gamma * &values_intrinsic[idx + 1] + &rewards_intrinsic
                - &values_intrinsic[idx];

            gae_intrinsic = delta + intrinsic_gl * &gae_intrinsic;
            adv_intrinsic.push(gae_intrinsic.copy());

            values_intrinsic[idx + 1] = td_value_intrinsic.copy();
            td_value_intrinsic = rewards_intrinsic + td_value_intrinsic * intrinsic_gamma;

            // Calculate the external rewards.
            let curr_step_rewards = 0.0;
            let delta =
                gamma * &values_external[idx + 1] + curr_step_rewards - &values_external[idx];

            gae_external = (delta + gl * &gae_external).clamp(-1.0, 1.0);
            adv_external.push(gae_external.copy());

            values_external[idx + 1] = td_value_external.copy();
            td_value_external = curr_step_rewards + td_value_external * gamma;

            td_error.push(
                f64::try_from((&td_value_external - &values_external[idx]).square()).unwrap(),
            );
        }

        values_external[0] = td_value_external;
        values_intrinsic[0] = td_value_intrinsic;

        adv_external.reverse();
        adv_intrinsic.reverse();
        td_error.reverse();

        let x = GAMECOUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if x % 1000 < 1 {
            // dbg!(x, &values_intrinsic, &adv_intrinsic);
        }

        let game_state = other.game_state.drain(..);
        let mut selected_policy = other.selected_policy.drain(..);
        let mut invalid_move_mask = other.invalid_move_mask.drain(..);
        let mut value_external = values_external.into_iter();
        let mut adv_external = adv_external.into_iter();
        let mut adv_intrinsic = adv_intrinsic.into_iter();
        let mut value_intrinsic = values_intrinsic.into_iter();
        let mut td_error = td_error.into_iter();

        let mut row_major_frames = Vec::new();
        for game_state in game_state {
            row_major_frames.push(Some(Frame {
                game_state,
                selected_policy: selected_policy.next().unwrap(),
                invalid_move_mask: invalid_move_mask.next().unwrap(),
                target_value: value_external.next().unwrap(),
                intrinsic_value: value_intrinsic.next().unwrap(),
                gae: adv_external.next().unwrap(),
                gae_intrinsic: adv_intrinsic.next().unwrap(),
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

            self.gae_intrinsic.push(frame.gae_intrinsic);

            self.target_value.push(frame.target_value);
            self.intrinsic_value.push(frame.intrinsic_value);
        }

        drop(selected_policy);
        drop(invalid_move_mask);
        other.clear();

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

    pub intrinsic_value: Vec<Tensor>,

    pub novelty: Vec<Tensor>,
}

impl SingleGame {
    pub fn clear(&mut self) {
        self.playing.take();
        self.game_state.clear();
        self.selected_policy.clear();
        self.invalid_move_mask.clear();
        self.value.clear();
        self.intrinsic_value.clear();
        self.novelty.clear();
    }

    pub fn len(&self) -> usize {
        self.game_state.len()
    }

    pub fn validate_buffers(&self) -> bool {
        let len = self.game_state.len();
        assert_eq!(len, self.selected_policy.len());
        assert_eq!(len, self.invalid_move_mask.len());
        assert_eq!(len, self.value.len());
        assert_eq!(len, self.intrinsic_value.len());
        assert_eq!(len, self.novelty.len());
        true
    }
}
