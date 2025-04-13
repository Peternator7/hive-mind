use core::f32;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use hive_engine::{
    game::Game,
    movement::Move,
    piece::{Color, Piece},
    position::Position,
};
use tch::{kind::FLOAT_CPU, Tensor};

use crate::hypers::{self, INPUT_ENCODED_DIMS};

pub fn length_masks() -> std::sync::MutexGuard<'static, Tensor> {
    static LENGTH_MASKS: OnceLock<Mutex<Tensor>> = OnceLock::new();

    LENGTH_MASKS
        .get_or_init(|| {
            let t = Tensor::from(1.0);

            Mutex::new(t)
        })
        .lock()
        .expect("woot")
}

pub trait PieceEncodable {
    const PLANES_PER_COLOR: usize;

    fn encode(&self, perspective: Color) -> usize;
}

impl PieceEncodable for Piece {
    const PLANES_PER_COLOR: usize = 11;

    fn encode(&self, perspective: Color) -> usize {
        let mut output: usize = match self.role {
            hive_engine::piece::Insect::QueenBee => 0,
            hive_engine::piece::Insect::SoldierAnt => 1,
            hive_engine::piece::Insect::Grasshopper => 4,
            hive_engine::piece::Insect::Beetle => 7,
            hive_engine::piece::Insect::Spider => 9,
        };

        output += self.id;
        if self.color != perspective {
            output += Self::PLANES_PER_COLOR
        }

        output
    }
}

pub fn translate_game_to_conv_tensor(game: &Game, perspective: Color) -> Tensor {
    let d = hive_engine::BOARD_SIZE as i64;

    let mut t = Tensor::zeros(&[hypers::INPUT_ENCODED_DIMS as i64, d, d], FLOAT_CPU);
    let b = game.board();

    let mut v = vec![0.0f32; hypers::INPUT_ENCODED_DIMS];
    let mut pieces = Vec::new();

    for pos in b.enumerate_all_pieces() {
        let mut height = 0;
        pieces.extend(b.enumerate_all_pieces_on_tile(pos));
        pieces.reverse();

        for piece in pieces.drain(..) {
            v[piece.encode(perspective)] = 1.0;

            if piece.is_beetle() {
                if height > 0 {
                    let mut idx = 2 * Piece::PLANES_PER_COLOR + height - 1;
                    if piece.color != perspective {
                        idx += 4;
                    }

                    v[idx] = 1.0;
                }

                height += 1;
            }
        }

        let _ = t.index_put_(
            &[
                None,
                Some(Tensor::from(pos.0 as i32)),
                Some(Tensor::from(pos.1 as i32)),
            ],
            &Tensor::from_slice(&v),
            false,
        );

        pieces.clear();
        v.fill(0.0);
    }

    t
}

pub fn translate_to_valid_absolute_moves_mask(
    game: &Game,
    valid_moves: &[Move],
    playing: Color,
    mask: &mut [bool],
) -> HashMap<usize, Move> {
    let mut map = HashMap::new();
    mask.fill(true);

    let dim = game.board().dims();
    for mv in valid_moves {
        let (plane, pos) = match mv {
            Move::MovePiece { piece, to, .. } => (piece.encode(playing), to),
            Move::PlacePiece { piece, position } => (piece.encode(playing), position),
            Move::Pass => unreachable!(),
        };

        let idx = (pos.1 as usize) + (dim * (pos.0 as usize)) + (dim * dim * plane);

        map.entry(idx).insert_entry(*mv);
        mask[idx] = false;
    }

    map
}

pub fn translate_to_valid_moves_mask(
    game: &Game,
    valid_moves: &[Move],
    playing: Color,
    mask: &mut [bool],
) -> HashMap<usize, Move> {
    let mut map = HashMap::new();
    mask.fill(true);

    fn inner(piece_idx: usize, adj_piece_idx: usize, relative_pos: usize) -> usize {
        // These make a 11 x 22 x 7 matrix.
        // 11 pieces can be moved
        // 22 pieces total on the board
        // 7 relative directions (6 hex directions + on top of)

        22 * 7 * piece_idx + 7 * adj_piece_idx + relative_pos
    }

    for mv in valid_moves {
        let (plane, pos) = match mv {
            Move::MovePiece { piece, to, .. } => (piece.encode(playing), to),
            Move::PlacePiece { piece, position } => (piece.encode(playing), position),
            Move::Pass => unreachable!(),
        };

        for piece in game.board().enumerate_all_pieces_on_tile(*pos) {
            let adj_piece_idx = piece.encode(playing);
            let relative_position = 6usize;
            let idx = inner(plane, adj_piece_idx, relative_position);
            mask[idx] = false;
            map.entry(idx).insert_entry(*mv);
        }

        for (direction, neighbor_pos) in pos.neighbors(game.board().dims() as u8) {
            for piece in game.board().enumerate_all_pieces_on_tile(neighbor_pos) {
                let adj_piece_idx = piece.encode(playing);
                let relative_position: usize = match direction {
                    hive_engine::position::Direction::TopRight => 0,
                    hive_engine::position::Direction::Right => 1,
                    hive_engine::position::Direction::BottomRight => 2,
                    hive_engine::position::Direction::BottomLeft => 3,
                    hive_engine::position::Direction::Left => 4,
                    hive_engine::position::Direction::TopLeft => 5,
                };

                let idx = inner(plane, adj_piece_idx, relative_position);
                mask[idx] = false;
                map.entry(idx).insert_entry(*mv);
            }
        }
    }

    map
}

pub fn translate_game_to_seq_tensor(game: &Game, perspective: Color) -> Tensor {
    let d = hive_engine::BOARD_SIZE as i64;

    let mut t = Tensor::zeros(&[hypers::INPUT_ENCODED_DIMS as i64, d * d], FLOAT_CPU);
    let b = game.board();

    let mut v = vec![0.0f32; hypers::INPUT_ENCODED_DIMS];
    let mut pieces = Vec::new();

    for pos in b.enumerate_all_pieces() {
        let mut height = 0;
        pieces.extend(b.enumerate_all_pieces_on_tile(pos));
        pieces.reverse();

        for piece in pieces.drain(..) {
            v[piece.encode(perspective)] = 1.0;

            if piece.is_beetle() {
                if height > 0 {
                    let mut idx = 2 * Piece::PLANES_PER_COLOR + height - 1;
                    if piece.color != perspective {
                        idx += 4;
                    }

                    v[idx] = 1.0;
                }

                height += 1;
            }
        }

        let _ = t.index_put_(
            &[
                None,
                Some(Tensor::from(pos.0 as i32)),
                Some(Tensor::from(pos.1 as i32)),
            ],
            &Tensor::from_slice(&v),
            false,
        );

        pieces.clear();
        v.fill(0.0);
    }

    t
}

pub fn translate_pieces_to_seq_tensor(
    pieces: &[(Position, Piece, usize)],
    perspective: Color,
) -> (Tensor, usize) {
    let mut t = Tensor::zeros(
        [
            hypers::MAX_SEQ_LENGTH,
            hypers::INPUT_ENCODED_DIMS as i64 + 6,
        ],
        tch::kind::FLOAT_CPU,
    );
    for (idx, (pos, piece, height)) in pieces.iter().enumerate() {
        let i = piece.encode(perspective);
        let (x, y, z) = pos.to_cube_coords();
        assert!(z <= 0);

        let emb = &mut [0.0; hypers::INPUT_ENCODED_DIMS + 6];

        emb[i] = 1.0;
        if *height > 0 {
            let mut idx = 2 * Piece::PLANES_PER_COLOR + height - 1;
            if piece.color != perspective {
                idx += 4;
            }

            emb[idx] = 1.0;
        }

        let x_encode = X_POSITION_ENCODINGS.with(|x_pos_encodings| x_pos_encodings[x as usize]);
        let y_encode = Y_POSITION_ENCODINGS.with(|y_pos_encodings| y_pos_encodings[y as usize]);
        let z_encode = Z_POSITION_ENCODINGS.with(|z_pos_encodings| z_pos_encodings[(-z) as usize]);

        emb[INPUT_ENCODED_DIMS + 0] = x_encode.0;
        emb[INPUT_ENCODED_DIMS + 1] = x_encode.1;
        emb[INPUT_ENCODED_DIMS + 2] = y_encode.0;
        emb[INPUT_ENCODED_DIMS + 3] = y_encode.1;
        emb[INPUT_ENCODED_DIMS + 4] = z_encode.0;
        emb[INPUT_ENCODED_DIMS + 5] = z_encode.1;

        let _ = t.index_put_(
            &[Some(Tensor::from(idx as i64))],
            &Tensor::from_slice(emb),
            false,
        );
    }

    (t, pieces.len())
}

thread_local! {
    static LENGTH_MASKS: Vec<Tensor> = {
        Vec::new()
    };

    static Z_POSITION_ENCODINGS: Vec<(f32,f32)> = {
        let s = hive_engine::BOARD_SIZE as usize;
        build_sin_cos_tensor(s, 3*s, 2 * s)
    };

    static Y_POSITION_ENCODINGS: Vec<(f32,f32)> = {
        let s = hive_engine::BOARD_SIZE as usize;
        build_sin_cos_tensor(s, 3*s, s)
    };

    static X_POSITION_ENCODINGS: Vec<(f32,f32)> = {
        let s = hive_engine::BOARD_SIZE as usize;
        build_sin_cos_tensor(s, 3*s, 0)
    };
}

pub fn build_sin_cos_tensor(
    board_size: usize,
    period: usize,
    phase_shift: usize,
) -> Vec<(f32, f32)> {
    let size = 2 * board_size + 1;

    let period = 2.0 * f32::consts::PI / period as f32;
    let phase_shift = period * phase_shift as f32;

    let mut output = Vec::new();
    for i in 0..size {
        output.push((
            f32::sin(period * i as f32 + phase_shift),
            f32::cos(period * i as f32 + phase_shift),
        ));
    }

    output
}
