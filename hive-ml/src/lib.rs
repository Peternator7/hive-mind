pub mod hypers;
pub mod frames;
pub mod model;
pub mod model2;
pub mod encode;
pub mod metrics;

pub trait TrainableFromFrames {
    fn extract_mask_tensor(&self, frames: &frames::MultipleGames) -> tch::Tensor;
    fn extract_action_tensor(&self, frames: &frames::MultipleGames) -> tch::Tensor;
    fn actor_critic(&self, game_state: tch::Tensor) -> (tch::Tensor, tch::Tensor);
}
