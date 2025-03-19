use crate::{piece::Piece, position::Position};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Move {
    MovePiece {
        piece: Piece,
        from: Position,
        to: Position,
    },
    PlacePiece {
        piece: Piece,
        position: Position,
    },
    Pass,
}
