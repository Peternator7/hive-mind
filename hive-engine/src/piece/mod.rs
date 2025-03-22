use std::fmt::{self, Display};

mod color;
mod insect;

pub use color::*;
pub use insect::*;

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Piece {
    pub role: Insect,
    pub color: Color,
    pub id: usize,
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

    pub fn from_char_pair(role: char, id: char) -> Option<Piece> {
        let id: usize = match id {
            '0' => 0,
            '1' => 1,
            '2' => 2,
            '3' => 3,
            _ => return None,
        };

        let output = match role {
            'q' => Piece {
                role: Insect::QueenBee,
                color: Color::White,
                id,
            },
            'a' => Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
                id,
            },
            'b' => Piece {
                role: Insect::Beetle,
                color: Color::White,
                id,
            },
            'g' => Piece {
                role: Insect::Grasshopper,
                color: Color::White,
                id,
            },
            's' => Piece {
                role: Insect::Spider,
                color: Color::White,
                id,
            },
            'Q' => Piece {
                role: Insect::QueenBee,
                color: Color::Black,
                id,
            },
            'A' => Piece {
                role: Insect::SoldierAnt,
                color: Color::Black,
                id,
            },
            'B' => Piece {
                role: Insect::Beetle,
                color: Color::Black,
                id,
            },
            'G' => Piece {
                role: Insect::Grasshopper,
                color: Color::Black,
                id,
            },
            'S' => Piece {
                role: Insect::Spider,
                color: Color::Black,
                id,
            },
            _ => return None,
        };

        return Some(output);
    }

    pub fn from_str(s: &str) -> Option<Piece> {
        let mut chars = s.chars();
        let ch = chars.next()?;
        let id = chars.next().unwrap_or('0');

        Self::from_char_pair(ch, id)
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = match self {
            Piece {
                role: Insect::Beetle,
                color: Color::White,
                ..
            } => 'b',
            Piece {
                role: Insect::Beetle,
                color: Color::Black,
                ..
            } => 'B',
            Piece {
                role: Insect::QueenBee,
                color: Color::White,
                ..
            } => 'q',
            Piece {
                role: Insect::QueenBee,
                color: Color::Black,
                ..
            } => 'Q',
            Piece {
                role: Insect::Spider,
                color: Color::White,
                ..
            } => 's',
            Piece {
                role: Insect::Spider,
                color: Color::Black,
                ..
            } => 'S',
            Piece {
                role: Insect::Grasshopper,
                color: Color::White,
                ..
            } => 'g',
            Piece {
                role: Insect::Grasshopper,
                color: Color::Black,
                ..
            } => 'G',
            Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
                ..
            } => 'a',
            Piece {
                role: Insect::SoldierAnt,
                color: Color::Black,
                ..
            } => 'A',
        };

        write!(f, "{}{}", c, self.id)
    }
}
