use std::fmt::{self, Display};

mod color;
mod insect;

pub use color::*;
pub use insect::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Piece {
    pub role: Insect,
    pub color: Color,
}

impl Piece {
    pub fn is_white_piece(&self) -> bool {
        matches!(self.color, Color::White)
    }

    pub fn is_black_piece(&self) -> bool {
        matches!(self.color, Color::Black)
    }

    pub fn is_beetle(&self) -> bool {
        matches!(self.role, Insect::Beetle)
    }

    pub fn is_queen(&self) -> bool {
        matches!(self.role, Insect::QueenBee)
    }

    pub fn from_char(ch: char) -> Option<Piece> {
        let output = match ch {
            'q' => Piece {
                role: Insect::QueenBee,
                color: Color::White,
            },
            'a' => Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
            },
            'b' => Piece {
                role: Insect::Beetle,
                color: Color::White,
            },
            'g' => Piece {
                role: Insect::Grasshopper,
                color: Color::White,
            },
            's' => Piece {
                role: Insect::Spider,
                color: Color::White,
            },
            'Q' => Piece {
                role: Insect::QueenBee,
                color: Color::Black,
            },
            'A' => Piece {
                role: Insect::SoldierAnt,
                color: Color::Black,
            },
            'B' => Piece {
                role: Insect::Beetle,
                color: Color::Black,
            },
            'G' => Piece {
                role: Insect::Grasshopper,
                color: Color::Black,
            },
            'S' => Piece {
                role: Insect::Spider,
                color: Color::Black,
            },
            _ => return None,
        };

        return Some(output);
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = match self {
            Piece {
                role: Insect::Beetle,
                color: Color::White,
            } => 'b',
            Piece {
                role: Insect::Beetle,
                color: Color::Black,
            } => 'B',
            Piece {
                role: Insect::QueenBee,
                color: Color::White,
            } => 'q',
            Piece {
                role: Insect::QueenBee,
                color: Color::Black,
            } => 'Q',
            Piece {
                role: Insect::Spider,
                color: Color::White,
            } => 's',
            Piece {
                role: Insect::Spider,
                color: Color::Black,
            } => 'S',
            Piece {
                role: Insect::Grasshopper,
                color: Color::White,
            } => 'g',
            Piece {
                role: Insect::Grasshopper,
                color: Color::Black,
            } => 'G',
            Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
            } => 'a',
            Piece {
                role: Insect::SoldierAnt,
                color: Color::Black,
            } => 'A',
        };

        write!(f, "{}", c)
    }
}
