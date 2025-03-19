pub mod agent;
pub mod ab_agent;
pub mod board;
pub mod error;
pub mod game;
pub mod hand;
pub mod movement;
pub mod piece;
pub mod position;

pub type Result<T> = std::result::Result<T, error::HiveError>;
