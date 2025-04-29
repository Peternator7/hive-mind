use hive_engine::{game::Game, piece::Color};

pub mod acc;
pub mod encode;
pub mod frames;
pub mod hypers;
pub mod metrics;
pub mod model;
pub mod model2;

pub trait Agent {
    fn set_color(&mut self, color: Color);

    fn select_move(&mut self, g: &Game);

    fn observe_final_state(&mut self, g: &Game);
}

pub struct RandomAgent{
    playing_as: Color,
}
