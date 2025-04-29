pub const INPUT_ENCODED_DIMS: usize = 2 * 11 + (2 * 4); // 11 pieces, 4 levels of beetle stacking.
                                                        // pub const OUTPUT_LENGTH: usize = BOARD_SIZE as usize * BOARD_SIZE as usize * 11;
pub const OUTPUT_LENGTH: usize = 11 * 22 * 7;
// pub const OUTPUT_LENGTH: usize = hive_engine::BOARD_SIZE as usize * hive_engine::BOARD_SIZE as usize * 11;

/// If the KL Loss gets high, we should stop the training
/// batch because we might be moving outside of the proximal
/// policy.
pub const CUTOFF_KL: f64 = 0.01; // 1.5 * 0.03;
pub const EPSILON: f64 = 0.20;

/// Number of games to run in parallel when training/testing.
pub const PARALLEL_GAMES: usize = 8;
pub const NUMBER_OF_MODELS: usize = 1;
pub const MODEL_NAMES: &'static [&'static str] = &[
    "model_a", "model_b", "model_c", "model_d", "model_e", "model_f", "model_g", "model_h",
];

pub const GAMES_PER_AI_SIMULATION: usize = 100;

/// The number of frames we're shooting for when generating training data in a
/// whole batch. It isn't an exact target.
pub const TARGET_FRAMES_PER_BATCH: usize = 16_000;

/// The number of turns after which we cancel a game.
pub const MAX_TURNS_PER_GAME: usize = 500;

/// The max number of frames from a game to include. We take the last n
/// frames under the assumption that frames towards the end of the game
/// are more likely to have affected the outcome than beginning frames.
/// This also prevents us from learning too much about any one trajectory.
pub const MAX_FRAMES_PER_GAME: usize = 60;
pub const MIN_FRAMES_PER_GAME: usize = 20;

/// Max number of iters we'll do on a single batch of data.
pub const TRAIN_ITERS_PER_BATCH: usize = 2;

/// The minibatch size for training.
pub const BATCH_SIZE: usize = 1024;

// pub const EPOCHS: usize = 251;
pub const EPOCHS: usize = 501;

/// We want to learn the policy at a slower rate than the
/// value because it's typically less stable.
pub const PI_LOSS_RATIO: f64 = 1.0;
pub const ENTROPY_LOSS_RATIO: f64 = 0.01;

// pub const EPOCHS_TO_USE_SPECIAL_LEARNING_RATE: usize = 3;
// pub const LEARNING_RATE_ON_FIRST_N_EPOCHS: f64 = 1e-3;
pub const INITIAL_LEARNING_RATE: f64 = 1e-4;
pub const LEARNING_RATE_DECREASE: f64 = INITIAL_LEARNING_RATE / EPOCHS as f64;

// pub const PENALIZE_TURNS_DISTANCE_FROM_END: isize = 60;
pub const PENALTY_FOR_MOVING: f64 = 0.01;
pub const PENALTY_FOR_TIMING_OUT: f64 = 0.500;
pub const APPROXIMATE_TURN_MEMORY: usize = 100;
pub const GAMMA: f64 = 1.0 - (1.0 / APPROXIMATE_TURN_MEMORY as f64);
pub const LAMBDA: f64 = 0.90;
pub const GRAD_CLIP: f64 = 0.50;
pub const INTRINSIC_ADV_SCALING: f64 = 0.10;

pub const MAX_SEQ_LENGTH: i64 = 2 * 11;

pub const STEP_BY_INGESTING_GAME: usize = 2;

pub const POLICY_PHASE_TRAIN_ITERS: usize = 2;
pub const AUXILIARY_PHASE_TRAIN_ITERS: usize = 10;
pub const AUXILIARY_PHASE_TRAIN_FREQUENCY: usize = 8;
pub const AUXILIARY_VALUE_SCALE: f64 = 1.0;
pub const AUXILIARY_BETA_CLONE: f64 = 0.10;
