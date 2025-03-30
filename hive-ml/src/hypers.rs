pub const INPUT_ENCODED_DIMS: usize = 2 * 11 + (2 * 4); // 11 pieces, 4 levels of beetle stacking.
pub const OUTPUT_LENGTH: usize = 11 * 22 * 7;

/// If the KL Loss gets high, we should stop the training
/// batch because we might be moving outside of the proximal
/// policy.
pub const CUTOFF_KL: f32 = 1.5 * 0.020;

/// Number of games to run in parallel when training/testing.
pub const PARALLEL_GAMES: usize = 8;

pub const GAMES_PER_AI_SIMULATION: usize = 100;

/// The number of frames we're shooting for when generating training data in a
/// whole batch. It isn't an exact target.
pub const TARGET_FRAMES_PER_BATCH: usize = 32_000;

/// The number of turns after which we cancel a game.
pub const MAX_TURNS_PER_GAME: usize = 2000;

/// The max number of frames from a game to include. We take the last n
/// frames under the assumption that frames towards the end of the game
/// are more likely to have affected the outcome than beginning frames.
/// This also prevents us from learning too much about any one trajectory.
pub const MAX_FRAMES_PER_GAME: usize = 100;

/// Max number of iters we'll do on a single batch of data.
pub const TRAIN_ITERS_PER_BATCH: usize = 50;

/// The minibatch size for training.
pub const BATCH_SIZE: usize = 128;

/// We want to learn the policy at a slower rate than the
/// value because it's typically less stable.
pub const PI_LOSS_RATIO: f64 = 0.1;

pub const LEARNING_RATE: f64 = 1e-3;

pub const GAMMA: f64 = 0.99;
pub const LAMBDA: f64 = 0.75;

pub const MAX_SEQ_LENGTH: i64 = 2 * 11;
