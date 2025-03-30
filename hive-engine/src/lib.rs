pub mod ab_agent;
pub mod board;
pub mod error;
pub mod game;
pub mod hand;
pub mod movement;
pub mod piece;
pub mod position;
pub mod utils;

pub type Result<T> = std::result::Result<T, error::HiveError>;

pub const BOARD_SIZE: u8 = 26;