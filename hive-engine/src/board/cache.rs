use crate::position::Position;

use super::tarjan::{self};

#[derive(Debug, Clone, Default)]
pub struct BoardCache {
    pub articulation_point_cache: Vec<ActuationPointIterStateCache>,
}

#[derive(Debug, Clone, Default)]
pub struct ActuationPointIterStateCache {
    pub state: Vec<(Position, usize)>,
    pub on_stack: Vec<tarjan::OnStackState>,
}
