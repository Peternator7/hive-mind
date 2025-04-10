pub const INPUT_ENCODED_DIMS: usize = 2 * 11 + (2 * 4); // 11 pieces, 4 levels of beetle stacking.
pub const OUTPUT_LENGTH: usize = 11 * 22 * 7;

/// If the KL Loss gets high, we should stop the training
/// batch because we might be moving outside of the proximal
/// policy.
pub const CUTOFF_KL: f32 = 0.05;

/// Number of games to run in parallel when training/testing.
pub const PARALLEL_GAMES: usize = 8;

pub const GAMES_PER_AI_SIMULATION: usize = 250;

/// The number of frames we're shooting for when generating training data in a
/// whole batch. It isn't an exact target.
pub const TARGET_FRAMES_PER_BATCH: usize = 36_000;

/// The number of turns after which we cancel a game.
pub const MAX_TURNS_PER_GAME: usize = 2000;

/// The max number of frames from a game to include. We take the last n
/// frames under the assumption that frames towards the end of the game
/// are more likely to have affected the outcome than beginning frames.
/// This also prevents us from learning too much about any one trajectory.
pub const MAX_FRAMES_PER_GAME: usize = 60;

/// Max number of iters we'll do on a single batch of data.
pub const TRAIN_ITERS_PER_BATCH: usize = 50;

/// The minibatch size for training.
pub const BATCH_SIZE: usize = 512;

/// We want to learn the policy at a slower rate than the
/// value because it's typically less stable.
pub const PI_LOSS_RATIO: f64 = 1.0;
pub const ENTROPY_LOSS_RATIO: f64 = 0.0002;
pub const MIN_ENTROPY_LOSS_RATIO: f64 = 0.0002;
pub const MAX_ENTROPY_LOSS_RATIO: f64 = 0.0100;

pub const INITIAL_LEARNING_RATE: f64 = 2e-4;
pub const MIN_LEARNING_RATE: f64 = 1e-7;

pub const APPROXIMATE_TURN_MEMORY: usize = 30;
pub const GAMMA: f64 = 1.0 - (1.0 / APPROXIMATE_TURN_MEMORY as f64);
pub const LAMBDA: f64 = 0.75;

pub const MAX_SEQ_LENGTH: i64 = 2 * 11;

pub const WIN_RATE_TO_FLIP_SIDES: f64 = 75.0;
pub const WIN_RATE_SMOOTHING_FACTOR: f64 = 0.667;
