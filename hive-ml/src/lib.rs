use hive_engine::{
    game::Game,
    piece::{Color, Piece},
};
use hypers::INPUT_ENCODED_DIMS;
use tch::{kind::FLOAT_CPU, Tensor};

pub mod hypers;
pub mod frames;

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

pub fn translate_to_tensor(game: &Game, perspective: Color) -> Tensor {
    let d = hive_engine::BOARD_SIZE as i64;

    let mut t = Tensor::zeros(&[INPUT_ENCODED_DIMS as i64, d, d], FLOAT_CPU);
    let b = game.board();

    let mut v = vec![0.0f32; INPUT_ENCODED_DIMS];
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
