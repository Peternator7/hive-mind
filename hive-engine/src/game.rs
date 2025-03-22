use std::hash::{DefaultHasher, Hash, Hasher};

use crate::{
    board::{frozen_map::FrozenMap, Board, PieceMovements},
    hand::Hand,
    movement::Move,
    piece::{Color, ColorMap, Insect, Piece},
    position::Position,
    Result,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GameWinner {
    Winner(Color),
    Tie,
}

#[derive(Debug, Clone)]
pub struct Game {
    transposition_key: u64,
    winner: Option<GameWinner>,
    board: Board,
    hands: ColorMap<Hand>,
    queen_location: ColorMap<Option<Position>>,
    turn: usize,
    to_play: Color,
    move_history: Vec<(Move, u64)>,
    frozenness: FrozenMap,
    restore_with_frozen_map: Option<usize>,
}

impl Game {
    pub fn new() -> Self {
        let board = Board::default();
        let frozenness = board.frozenness();
        Self {
            board,
            frozenness,
            winner: None,
            restore_with_frozen_map: None,
            hands: ColorMap::default(),
            queen_location: ColorMap::default(),
            turn: 1,
            to_play: Color::White,
            move_history: Vec::new(),
            transposition_key: 0,
        }
    }

    pub fn dims(&self) -> usize {
        self.board.dims()
    }

    pub fn is_game_is_over(&self) -> Option<GameWinner> {
        self.winner
    }

    pub fn turn(&self) -> usize {
        self.turn
    }

    pub fn to_play(&self) -> Color {
        self.to_play
    }

    pub fn hand(&self, color: Color) -> &Hand {
        self.hands.get(color)
    }

    pub fn queen_pos(&self, color: Color) -> Option<Position> {
        *self.queen_location.get(color)
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn set_frozenness_checkpoint(&mut self) {
        self.board.clone_frozenness(&mut self.frozenness);
        self.restore_with_frozen_map = Some(self.move_history.len());
    }

    pub fn clear_frozenness_checkpoint(&mut self) {
        self.restore_with_frozen_map = None;
    }

    pub fn transposition_key(&self) -> u64 {
        self.transposition_key
    }

    pub fn enumerate_playable_pieces_from_hand<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = Piece> + 'a {
        Insect::iter()
            .filter_map(move |insect| {
                self.hands
                    .get(color)
                    .next_insect_id(insect)
                    .map(|id| (id, insect))
            })
            .map(move |(id, insect)| Piece {
                role: insect,
                id,
                color,
            })
    }

    pub fn is_repeat_move(&self, mv: Move, depth: usize) -> bool {
        for (h_mv, h_key) in self.move_history.iter().rev().take(depth) {
            if self.transposition_key == *h_key && mv == *h_mv {
                return true;
            }
        }

        false
    }

    pub fn enumerate_playable_positions<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = Position> + 'a {
        self.board.enumerate_playable_positions(color)
    }

    pub fn enumerate_free_pieces<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Piece, Position)> + 'a {
        self.board.enumerate_free_pieces(color)
    }

    pub fn enumerate_valid_moves<'a>(&'a self, pos: Position) -> Result<PieceMovements<'a>> {
        self.board.enumerate_valid_moves(pos)
    }

    pub fn load_all_potential_moves(&self, dst: &mut Vec<Move>) -> Result<()> {
        let to_play = self.to_play();
        let turn = self.turn();

        let queen_in_hand = self.hand(to_play).has_queen();
        let can_move = !queen_in_hand;
        let can_play_queen = turn > 1;
        let must_play_queen = turn == 4 && queen_in_hand;

        for piece in self.enumerate_playable_pieces_from_hand(to_play) {
            if !can_play_queen && piece.is_queen() {
                continue;
            }

            if must_play_queen && !piece.is_queen() {
                continue;
            }

            for position in self.enumerate_playable_positions(to_play) {
                let mv = Move::PlacePiece { piece, position };
                dst.push(mv)
            }
        }

        if !can_move {
            return Ok(());
        }

        for (piece, from) in self.enumerate_free_pieces(to_play) {
            for to in self.enumerate_valid_moves(from)? {
                let mv = Move::MovePiece { piece, from, to };
                dst.push(mv);
            }
        }

        Ok(())
    }

    pub fn make_move(&mut self, mv: Move) -> Result<()> {
        debug_assert!(self.winner.is_none());

        let lifetime_buffer_1;
        let lifetime_buffer_2;

        let things_to_hash: &[(Piece, Position, usize)];

        match mv {
            Move::PlacePiece { piece, position } => {
                self.board.place_piece_in_empty_cell(piece, position)?;
                self.hands
                    .get_mut(piece.color)
                    .pop_tile(piece.role)
                    .unwrap();
                let height = self.board.height(position);

                if piece.is_queen() {
                    *self.queen_location.get_mut(piece.color) = Some(position);
                }

                lifetime_buffer_1 = [(piece, position, height)];
                things_to_hash = &lifetime_buffer_1;
            }
            Move::MovePiece { piece, from, to } => {
                debug_assert_eq!(Some(piece), self.board[from].piece.get());

                let from_height = self.board.height(from);
                self.board.move_piece(from, to)?;
                let to_height = self.board.height(to);

                if piece.is_queen() {
                    *self.queen_location.get_mut(piece.color) = Some(to);
                }

                lifetime_buffer_2 = [(piece, from, from_height), (piece, to, to_height)];
                things_to_hash = &lifetime_buffer_2;
            }
            Move::Pass => {
                things_to_hash = &[];
            }
        }

        let transposition_key = self.transposition_key;
        for tuple in things_to_hash {
            let mut hasher = DefaultHasher::new();
            tuple.hash(&mut hasher);
            let output = hasher.finish();
            self.transposition_key ^= output;
        }

        if let Some(pos) = self.queen_location.black() {
            if self.board[*pos].adjacent_pieces() == 6 {
                self.winner = Some(GameWinner::Winner(Color::White));
            }
        }

        if let Some(pos) = self.queen_location.white() {
            if self.board[*pos].adjacent_pieces() == 6 {
                if self.winner.is_some() {
                    self.winner = Some(GameWinner::Tie);
                } else {
                    self.winner = Some(GameWinner::Winner(Color::Black));
                }
            }
        }

        self.move_history.push((mv, transposition_key));
        self.to_play = self.to_play.opposing();
        if self.to_play == Color::White {
            self.turn += 1;
        }

        Ok(())
    }

    pub fn unmake_move(&mut self) -> Result<Move> {
        let Some((last_move, _)) = self.move_history.pop() else {
            panic!();
        };

        let lifetime_buffer_1;
        let lifetime_buffer_2;

        let things_to_hash: &[(Piece, Position, usize)];
        match last_move {
            Move::PlacePiece { piece, position } => {
                let height = self.board.height(position);
                let removed = if self.restore_with_frozen_map == Some(self.move_history.len()) {
                    self.board
                        .remove_piece_and_restore_frozen_map(position, &self.frozenness)?
                } else {
                    self.board.remove_piece(position)?
                };

                debug_assert_eq!(piece, removed);
                self.hands.get_mut(piece.color).push_tile(piece.role);

                if piece.is_queen() {
                    *self.queen_location.get_mut(piece.color) = None;
                }

                lifetime_buffer_1 = [(piece, position, height)];
                things_to_hash = &lifetime_buffer_1;
            }
            Move::MovePiece { piece, from, to } => {
                debug_assert_eq!(Some(piece), self.board[to].piece.get());

                let to_height = self.board.height(to);

                if self.restore_with_frozen_map == Some(self.move_history.len()) {
                    self.board
                        .move_piece_and_restore_frozen_map(to, from, &self.frozenness)?;
                } else {
                    self.board.move_piece(to, from)?;
                }

                let from_height = self.board.height(from);

                if piece.is_queen() {
                    *self.queen_location.get_mut(piece.color) = Some(from);
                }

                lifetime_buffer_2 = [(piece, from, from_height), (piece, to, to_height)];
                things_to_hash = &lifetime_buffer_2;
            }
            Move::Pass => {
                things_to_hash = &[];
            }
        }

        for tuple in things_to_hash {
            let mut hasher = DefaultHasher::new();
            tuple.hash(&mut hasher);
            let output = hasher.finish();
            self.transposition_key ^= output;
        }

        self.winner = None;
        self.to_play = self.to_play.opposing();
        if self.to_play == Color::Black {
            self.turn -= 1;
        }

        Ok(last_move)
    }

    pub fn calculate_score(&self) -> f32 {
        self.board.score()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        movement::Move,
        piece::{Color, Insect, Piece},
        position::Position,
    };

    use super::Game;

    #[test]
    fn simple_transposition_key_checks() {
        let mut g = Game::new();

        assert_eq!(0, g.transposition_key());
        g.make_move(Move::PlacePiece {
            piece: Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
                id: 0,
            },
            position: Position(16, 16),
        })
        .unwrap();

        let key_after_first_play = g.transposition_key();

        g.make_move(Move::PlacePiece {
            piece: Piece {
                role: Insect::Beetle,
                color: Color::Black,
                id: 0,
            },
            position: Position(16, 15),
        })
        .unwrap();

        let key_after_second_play = g.transposition_key();

        g.make_move(Move::MovePiece {
            piece: Piece {
                role: Insect::SoldierAnt,
                color: Color::White,
                id: 0,
            },
            from: Position(16, 16),
            to: Position(16, 14),
        })
        .unwrap();

        let key_after_third_play = g.transposition_key();

        g.make_move(Move::MovePiece {
            piece: Piece {
                role: Insect::Beetle,
                color: Color::Black,
                id: 0,
            },
            from: Position(16, 15),
            to: Position(16, 14),
        })
        .unwrap();

        g.unmake_move().unwrap();

        assert_eq!(key_after_third_play, g.transposition_key());

        g.unmake_move().unwrap();
        assert_eq!(key_after_second_play, g.transposition_key());

        g.unmake_move().unwrap();
        assert_eq!(key_after_first_play, g.transposition_key());

        g.unmake_move().unwrap();
        assert_eq!(0, g.transposition_key());
    }
}
