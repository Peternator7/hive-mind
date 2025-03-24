pub const INPUT_ENCODED_DIMS: usize = 2 * 11 + (2 * 4); // 11 pieces, 4 levels of beetle stacking.
pub const OUTPUT_LENGTH: usize =
    11 * hive_engine::BOARD_SIZE as usize * hive_engine::BOARD_SIZE as usize;

pub const TRAIN_ITERS_PER_BATCH: usize = 50;
pub const CUTOFF_KL: f32 = 1.5 * 0.025;
pub const PARALLEL_GAMES: usize = 8;
pub const GAMES_PER_BATCH: usize = 16;
pub const TARGET_FRAMES_PER_BATCH: usize = 28_000;
pub const GAMES_PER_AI_SIMULATION: usize = 100;
pub const MAX_TURNS_PER_GAME: usize = 500;
pub const BATCH_SIZE: usize = 128;

pub const GAMMA: f64 = 0.99;
pub const LAMBDA: f64 = 0.95;
