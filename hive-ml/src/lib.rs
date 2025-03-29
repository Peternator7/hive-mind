use std::sync::{Mutex, OnceLock};

use hive_engine::{
    game::Game,
    piece::{Color, Piece},
};
use hypers::INPUT_ENCODED_DIMS;
use tch::{kind::FLOAT_CPU, Tensor};

pub mod hypers;
pub mod frames;
pub mod model;
pub mod model2;
pub mod encode;
