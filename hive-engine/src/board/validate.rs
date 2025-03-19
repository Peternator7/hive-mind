use std::collections::{HashSet, VecDeque};

use crate::{
    error::{HiveError, ValidationError},
    piece::Color,
};

use super::{Board, TilesIter};

impl Board {
    pub fn validate_board(&self) -> std::result::Result<(), HiveError> {
        let mut white_pieces_observed = 0;
        let mut black_pieces_observed = 0;
        let mut number_of_hives = 0;

        let mut playable_tiles = HashSet::new();
        if self.pieces_on_board.get() > 0 {
            playable_tiles.extend(TilesIter {
                board: self,
                current: self.first_playable_tile.get(),
            });
        }

        let mut seen = HashSet::new();
        let mut needs_visit = VecDeque::default();

        for (idx, _) in self.grid.iter().enumerate() {
            let idx = Board::convert_idx_to_coords(idx, self.dim);
            if seen.contains(&idx) {
                continue;
            }

            let cell_idx = &self[idx];
            let mut adjacent_white_pieces = 0;
            let mut adjacent_black_pieces = 0;
            for (_, neighbor_pos) in idx.neighbors(self.dim) {
                let neighbor = &self[neighbor_pos];
                let Some(neighbor_piece) = neighbor.piece.get() else {
                    continue;
                };

                if neighbor_piece.is_black_piece() {
                    adjacent_black_pieces += 1;
                }

                if neighbor_piece.is_white_piece() {
                    adjacent_white_pieces += 1;
                }
            }

            if adjacent_black_pieces != cell_idx.adjacent_pieces.black().get()
                || adjacent_white_pieces != cell_idx.adjacent_pieces.white().get()
            {
                return Err(HiveError::from(
                    ValidationError::TileAdjacencyCountsAreWrong,
                ));
            }

            if cell_idx.piece.get().is_none() {
                let cell_is_maybe_playable = cell_idx.adjacent_pieces() > 0;
                let cell_is_in_playable_list = playable_tiles.remove(&idx);

                // If the tile is empty, check if it's in the playable tiles list.
                if cell_is_maybe_playable != cell_is_in_playable_list {
                    return Err(HiveError::from(
                        ValidationError::AvailableTilesAreMissingFromLL,
                    ));
                }

                continue;
            }

            needs_visit.push_back(idx);
            seen.insert(idx);

            number_of_hives += 1;
            if number_of_hives > 1 {
                return Err(HiveError::from(ValidationError::MultipleHives));
            }

            while let Some(pos) = needs_visit.pop_front() {
                let cell = &self[pos];
                let piece = cell.piece.get().expect("only inspecting occupied cells");

                if piece.is_white_piece() {
                    white_pieces_observed += 1;
                }

                if piece.is_black_piece() {
                    black_pieces_observed += 1;
                }

                for (_, neighbor_pos) in pos.neighbors(self.dim) {
                    let neighbor = &self[neighbor_pos];
                    let Some(..) = neighbor.piece.get() else {
                        continue;
                    };

                    // If the neighbor is an occupied piece that we haven't seen,
                    // add it to the queue.
                    if !seen.contains(&neighbor_pos) {
                        needs_visit.push_back(neighbor_pos);
                        seen.insert(neighbor_pos);
                    }
                }
            }
        }

        let white_pieces_expected = self.enumerate_pieces(Color::White).count();
        let black_pieces_expected = self.enumerate_pieces(Color::Black).count();

        if white_pieces_observed != white_pieces_expected {
            return Err(HiveError::from(
                ValidationError::WhitePiecesAreMissingFromLL,
            ));
        }

        if black_pieces_observed != black_pieces_expected {
            return Err(HiveError::from(
                ValidationError::BlackPiecesAreMissingFromLL,
            ));
        }

        if !playable_tiles.is_empty() {
            return Err(HiveError::from(
                ValidationError::AvailableTilesAreMissingFromLL,
            ));
        }

        Ok(())
    }
}
