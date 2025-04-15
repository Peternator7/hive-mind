pub const INPUT_ENCODED_DIMS: usize = 2 * 11 + (2 * 4); // 11 pieces, 4 levels of beetle stacking.
                                                        // pub const OUTPUT_LENGTH: usize = BOARD_SIZE as usize * BOARD_SIZE as usize * 11;
pub const OUTPUT_LENGTH: usize = 11 * 22 * 7;

/// If the KL Loss gets high, we should stop the training
/// batch because we might be moving outside of the proximal
/// policy.
pub const CUTOFF_KL: f32 = 1.5 * 0.20;
pub const EPSILON: f64 = 0.20;

/// Number of games to run in parallel when training/testing.
pub const PARALLEL_GAMES: usize = 8;
pub const NUMBER_OF_MODELS: usize = 1;
pub const MODEL_NAMES: &'static [&'static str] = &[
    "model_a",
    "model_b",
    "model_c",
    "model_d",
    "model_e",
    "model_f",
    "model_g",
    "model_h",
];

pub const GAMES_PER_AI_SIMULATION: usize = 100;

/// The number of frames we're shooting for when generating training data in a
/// whole batch. It isn't an exact target.
pub const TARGET_FRAMES_PER_BATCH: usize = 18_000;

/// The number of turns after which we cancel a game.
pub const MAX_TURNS_PER_GAME: usize = 2000;

/// The max number of frames from a game to include. We take the last n
/// frames under the assumption that frames towards the end of the game
/// are more likely to have affected the outcome than beginning frames.
/// This also prevents us from learning too much about any one trajectory.
pub const MAX_FRAMES_PER_GAME: usize = 75;

/// Max number of iters we'll do on a single batch of data.
pub const TRAIN_ITERS_PER_BATCH: usize = 16;

/// The minibatch size for training.
pub const BATCH_SIZE: usize = 1024;

/// We want to learn the policy at a slower rate than the
/// value because it's typically less stable.
pub const PI_LOSS_RATIO: f64 = 1.0;
// pub const ENTROPY_LOSS_RATIO: f64 = 0.01;
pub const ENTROPY_LOSS_RATIO: f64 = 0.00;

// pub const EPOCHS_TO_USE_SPECIAL_LEARNING_RATE: usize = 3;
// pub const LEARNING_RATE_ON_FIRST_N_EPOCHS: f64 = 1e-3;
pub const INITIAL_LEARNING_RATE: f64 = 3e-4;
pub const MIN_LEARNING_RATE: f64 = 1e-7;
pub const LEARNING_RATE_DECAY: f64 = 0.98;

// pub const PENALIZE_TURNS_DISTANCE_FROM_END: isize = 60;
pub const APPROXIMATE_TURN_MEMORY: usize = 50;
pub const GAMMA: f64 = 1.0 - (1.0 / APPROXIMATE_TURN_MEMORY as f64);
pub const LAMBDA: f64 = 0.96;

pub const MAX_SEQ_LENGTH: i64 = 2 * 11;
