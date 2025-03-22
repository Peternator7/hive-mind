use std::{fmt::Display, num::NonZeroUsize};

use slab::Slab;

use crate::{
    board::CoveredPiece,
    error::{DeserializationError, HiveError},
    piece::Piece,
    position::Position,
};

use super::Board;

pub struct PrettyBoard<'a>(&'a Board);

pub struct DenseBoard<'a>(&'a Board);

impl Board {
    pub fn dense_format(&self) -> DenseBoard {
        DenseBoard(self)
    }

    pub fn pretty_format(&self) -> PrettyBoard {
        PrettyBoard(self)
    }

    pub fn from_dense_repr(text: &str, dim: u8) -> crate::Result<Self> {
        let mut output = Board::new(dim);
        let mut idx: usize = 0;
        let mut num_zeros = 0;
        let mut processing_stacked_beetle = false;
        let mut chars = text.chars();
        while let Some(ch) = chars.next() {
            match ch {
                '[' => {
                    processing_stacked_beetle = true;
                }
                ']' => {
                    processing_stacked_beetle = false;
                    idx += 1;
                }
                _ if ch.is_ascii_digit() => {
                    let digit = ch.to_digit(10).expect("checked in match arm");
                    num_zeros *= 10;
                    num_zeros += digit as usize;
                }
                _ => {
                    if num_zeros > 0 {
                        idx += num_zeros;
                        num_zeros = 0;
                    }

                    let Some(id) = chars.next() else {
                        return Err(HiveError::DeserializationError(
                            DeserializationError::EndOfStream,
                        ));
                    };

                    let Some(p) = Piece::from_char_pair(ch, id) else {
                        return Err(HiveError::DeserializationError(
                            DeserializationError::InvalidChar(ch, id),
                        ));
                    };

                    output
                        .internal_place_piece(
                            p,
                            Board::convert_idx_to_coords(idx, dim),
                            false,
                            p.is_beetle(),
                        )
                        .expect("yolo");

                    output.pieces_on_board.set(output.pieces_on_board.get() + 1);

                    if !processing_stacked_beetle {
                        idx += 1;
                    }
                }
            }
        }

        output.recalculate_frozen_cells();
        Ok(output)
    }

    pub fn from_string_repr(board: &str, validate: bool) -> crate::Result<Board> {
        let lines = board
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        let mut board = Board::new(lines.len() as u8);

        for (idx, line) in lines.iter().enumerate() {
            for (col, value) in line.split(' ').enumerate() {
                if value == "." {
                    continue;
                }

                let Some(piece) = Piece::from_str(value) else {
                    let mut chars = value.chars();
                    let c1 = chars.next();
                    let mut c2 = ' ';
                    if c1.is_some() {
                        c2 = chars.next().unwrap_or('?');
                    }

                    return Err(HiveError::DeserializationError(
                        DeserializationError::InvalidChar(c1.unwrap_or('?'), c2),
                    ));
                };

                board.internal_place_piece(
                    piece,
                    Position(idx as u8, col as u8),
                    false,
                    piece.is_beetle(),
                )?;

                board.pieces_on_board.set(board.pieces_on_board.get() + 1);
            }
        }

        board.recalculate_frozen_cells();
        if validate {
            board.validate_board()?;
        }

        Ok(board)
    }
}

impl<'a> Display for DenseBoard<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let board = self.0;
        let mut running_zeros = 0;
        for cell in board.grid.iter() {
            let Some(piece) = cell.piece.get() else {
                running_zeros += 1;
                continue;
            };

            if running_zeros > 0 {
                write!(f, "{}", running_zeros)?;
                running_zeros = 0;
            }

            let mut has_covered = false;
            if piece.is_beetle() {
                if let Some(v) = cell.covered_piece.get() {
                    fn print_ll_in_reverse(
                        f: &mut std::fmt::Formatter<'_>,
                        slab: &Slab<CoveredPiece>,
                        curr: NonZeroUsize,
                    ) -> std::fmt::Result {
                        let curr = &slab[curr.get()];
                        if let Some(next) = curr.next {
                            print_ll_in_reverse(f, slab, next)?;
                        }

                        write!(f, "{}", curr.piece)?;
                        Ok(())
                    }

                    has_covered = true;
                    write!(f, "[")?;

                    print_ll_in_reverse(f, &board.covered_pieces_two, v)?;
                }
            }

            write!(f, "{}", piece)?;
            if has_covered {
                write!(f, "]")?;
            }
        }

        Ok(())
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pretty_format())
    }
}

impl<'a> Display for PrettyBoard<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let board = self.0;
        let dim = board.dim as usize;

        for x in 0..dim {
            for y in 0..dim {
                let cell = &board[Position(x as u8, y as u8)];
                match cell.piece.get() {
                    Some(piece) => write!(f, "{}", piece)?,
                    None => write!(f, ".")?,
                }

                if y + 1 < dim {
                    write!(f, " ")?;
                }
            }

            if x + 1 < dim {
                writeln!(f)?;
                for _ in 0..(x + 1) {
                    write!(f, " ")?;
                }
            }
        }

        Ok(())
    }
}
