#![allow(dead_code)]

mod cache;
pub mod draw;
pub mod frozen_map;
pub mod hypers;
pub mod movement;
mod repr;
mod tarjan;
pub mod validate;

use std::{
    cell::{Cell, RefCell},
    hash::Hash,
    num::NonZeroUsize,
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

use cache::BoardCache;
use frozen_map::FrozenMap;
use slab::Slab;

use crate::{
    error::{self, HiveError},
    piece::{Color, ColorMap, Insect, Piece},
    position::Position,
    Result,
};

const BOARD_SIZE: u8 = 32;
pub static RECALCULATE_SKIPPED: AtomicUsize = AtomicUsize::new(0);
pub static RECALCULATE_REQUIRED: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct GridCell {
    pub piece: Cell<Option<Piece>>,
    pub adjacent_pieces: ColorMap<Cell<u8>>,
    next_piece: Cell<Option<NonZeroUsize>>,
    prev_piece: Cell<Option<NonZeroUsize>>,
    covered_piece: Cell<Option<NonZeroUsize>>,
}

impl GridCell {
    pub fn adjacent_pieces(&self) -> u8 {
        self.adjacent_pieces.white().get() + self.adjacent_pieces.black().get()
    }
}

#[derive(Copy, Clone, Debug)]
struct CoveredPiece {
    piece: Piece,
    next: Option<NonZeroUsize>,
}

#[derive(Debug, Clone)]
pub struct Board {
    dim: u8,
    grid: Vec<GridCell>,
    frozen_map: FrozenMap,
    covered_pieces_two: Slab<CoveredPiece>,
    first_white_piece: Cell<Option<NonZeroUsize>>,
    first_black_piece: Cell<Option<NonZeroUsize>>,
    first_playable_tile: Cell<Option<NonZeroUsize>>,
    cache: RefCell<BoardCache>,
    pieces_on_board: Cell<usize>,
    needs_frozen_refresher: Cell<bool>,
}

impl Index<Position> for Board {
    type Output = GridCell;

    fn index(&self, index: Position) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl Board {
    pub fn new(dim: u8) -> Board {
        // The dumb trick here is that we block out the zero index so that all
        // the rest of the indexes we get will be > than 0 and we can use NonZeroUsize for.
        let covered_pieces_two = [(
            0,
            CoveredPiece {
                piece: Piece {
                    role: Insect::Beetle,
                    color: Color::White,
                },
                next: None,
            },
        )]
        .into_iter()
        .collect::<Slab<_>>();

        Board {
            grid: vec![Default::default(); dim as usize * dim as usize],
            frozen_map: FrozenMap::new(dim as usize * dim as usize),
            dim,
            covered_pieces_two,
            cache: Default::default(),
            first_black_piece: Default::default(),
            first_white_piece: Default::default(),
            first_playable_tile: Default::default(),
            pieces_on_board: Default::default(),
            needs_frozen_refresher: Cell::new(false),
        }
    }

    pub fn frozenness(&self) -> FrozenMap {
        self.frozen_map.clone()
    }

    pub fn clone_frozenness(&self, dst: &mut FrozenMap) {
        dst.clone_from(&self.frozen_map);
    }

    pub fn get(&self, pos: Position) -> Option<&GridCell> {
        let Position(x, y) = pos;
        let idx = Board::convert_coords_to_idx(x, y, self.dim);
        self.grid.get(idx)
    }

    pub fn height(&self, pos: Position) -> usize {
        let cell = &self[pos];
        if cell.piece.get().is_none() {
            return 0;
        }

        let mut output = 1;
        let mut it = cell.covered_piece.get();
        while let Some(curr) = it {
            it = self.covered_pieces_two[curr.get()].next;
            output += 1;
        }

        output
    }

    pub fn remove_piece(&mut self, pos: Position) -> Result<Piece> {
        let output = self.internal_remove_piece(pos)?;
        self.pieces_on_board.set(self.pieces_on_board.get() - 1);
        self.recalculate_frozen_cells();
        Ok(output)
    }

    pub fn remove_piece_and_restore_frozen_map(
        &mut self,
        pos: Position,
        map: &FrozenMap,
    ) -> Result<Piece> {
        let output = self.internal_remove_piece(pos)?;
        self.pieces_on_board.set(self.pieces_on_board.get() - 1);
        self.frozen_map.clone_from(map);
        self.needs_frozen_refresher.set(false);
        Ok(output)
    }

    pub fn place_piece_in_empty_cell(&mut self, piece: Piece, pos: Position) -> Result<()> {
        self.internal_place_piece(piece, pos, true, false)?;
        self.pieces_on_board.set(self.pieces_on_board.get() + 1);
        self.recalculate_frozen_cells();
        Ok(())
    }

    pub fn move_piece(&mut self, from: Position, to: Position) -> Result<()> {
        let piece = self.internal_remove_piece(from)?;
        self.internal_place_piece(piece, to, false, piece.is_beetle())?;
        self.recalculate_frozen_cells();
        Ok(())
    }

    pub fn move_piece_and_restore_frozen_map(
        &mut self,
        from: Position,
        to: Position,
        map: &FrozenMap,
    ) -> Result<()> {
        let piece = self.internal_remove_piece(from)?;
        self.internal_place_piece(piece, to, false, piece.is_beetle())?;
        self.frozen_map.clone_from(map);
        self.needs_frozen_refresher.set(false);
        Ok(())
    }

    pub fn force_recalculate_frozen_cells(&self) {
        self.needs_frozen_refresher.set(true);
        self.recalculate_frozen_cells();
    }

    fn recalculate_frozen_cells(&self) {
        if !self.needs_frozen_refresher.get() {
            RECALCULATE_SKIPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }

        RECALCULATE_REQUIRED.fetch_add(1, Ordering::Relaxed);

        let start_pos = self
            .first_black_piece
            .get()
            .or(self.first_white_piece.get());

        let Some(start_pos) = start_pos else {
            // If there's no pieces on the board, none of them will be frozen :)
            return;
        };

        self.frozen_map.clear_all();

        let start_pos = Board::convert_idx_to_coords(start_pos.get(), self.dim);
        for pos in tarjan::ActuationPointIter::new(start_pos, self) {
            let cell = &self[pos];
            let idx = Board::convert_coords_to_idx(pos.0, pos.1, self.dim);
            debug_assert!(cell.piece.get().is_some());

            if self.frozen_map.get(idx) {
                continue;
            }

            if cell.adjacent_pieces() > 1 {
                self.frozen_map.set(idx);
            }
        }

        self.needs_frozen_refresher.set(false);
    }

    fn internal_place_piece(
        &mut self,
        piece: Piece,
        pos: Position,
        validate_position: bool,
        allow_beetle_move: bool,
    ) -> Result<()> {
        let idx = Board::convert_coords_to_idx(pos.0, pos.1, self.dim);
        if idx == 0 {
            return Err(HiveError::ZeroPositionIsUnplayable);
        }

        let Some(cell) = self.grid.get(idx) else {
            return Err(HiveError::PositionOutOfBounds);
        };

        let prev_piece = cell.piece.get();
        if validate_position {
            // We can't place a piece on top of another piece in most cases.
            if prev_piece.is_some() && !allow_beetle_move {
                return Err(HiveError::InvalidPlacementLocation(
                    error::InvalidPlacementLocation::PositionOccupied,
                ));
            };

            let same_color_piece_count = cell.adjacent_pieces.get(piece.color).get();
            let opposite_color_piece_count = cell.adjacent_pieces.get(piece.color.opposing()).get();

            match self.pieces_on_board.get() {
                // The first piece has no placement constraints
                0 => {}
                // The second piece must be adjacent to the first.
                1 if opposite_color_piece_count == 1 => {}
                1 => {
                    return Err(HiveError::InvalidPlacementLocation(
                        error::InvalidPlacementLocation::SecondPieceMustBeAdjacentToFirst,
                    ))
                }
                // Additional pieces must be placed adjacent to pieces of the same color
                // and not adjacent to any pieces of the opposite color.
                _ if opposite_color_piece_count > 0 => {
                    return Err(HiveError::InvalidPlacementLocation(
                        error::InvalidPlacementLocation::AdjacentToOppositeColor,
                    ))
                }
                _ if same_color_piece_count == 0 => {
                    return Err(HiveError::InvalidPlacementLocation(
                        error::InvalidPlacementLocation::NotAdjacentToOwnColor,
                    ))
                }
                _ => {}
            };
        }

        let mut needs_frozen_refresh = true;
        let mut can_maybe_skip_refresh = cell.adjacent_pieces() == 1;
        if prev_piece.is_some() {
            // If we're covering up another piece, then we don't need to recompute the "frozen" status of the tiles.
            needs_frozen_refresh = false;
            can_maybe_skip_refresh = false;
        }

        let curr_head = NonZeroUsize::new(idx);
        let pred_head = cell.prev_piece.take();
        let succ_head = cell.next_piece.take();
        let prev_list_head;
        if let Some(prev_piece) = prev_piece {
            assert!(piece.is_beetle());
            let idx = self.covered_pieces_two.insert(CoveredPiece {
                piece: prev_piece,
                next: cell.covered_piece.get(),
            });

            debug_assert!(idx > 0);
            cell.covered_piece.set(NonZeroUsize::new(idx));
            prev_list_head = self.get_ll_by_color(prev_piece.color);
        } else {
            // The cell is empty, but it won't be after this method exits. We need to remove it from the list
            // of empty/playable cells.
            prev_list_head = &self.first_playable_tile;
        }

        self.remove_from_linked_list(curr_head, pred_head, succ_head, prev_list_head);

        // Add the piece to the list of white/black pieces.
        cell.piece.set(Some(piece));
        let new_list_head = self.get_ll_by_color(piece.color);

        // Add the tile to the list of white/black pieces.
        self.add_to_linked_list(cell, curr_head, new_list_head);

        // Remove the tile from the list of empty tiles.
        for (_, neighbor_pos) in pos.neighbors(self.dim) {
            let neighbor = self
                .get(neighbor_pos)
                .expect("should be internally consistent");

            let adjacent_piece_counter = neighbor.adjacent_pieces.get(piece.color);
            adjacent_piece_counter.set(adjacent_piece_counter.get() + 1);
            if let Some(prev_color) = prev_piece.map(|p| p.color) {
                let previous_piece_counter = neighbor.adjacent_pieces.get(prev_color);
                previous_piece_counter.set(previous_piece_counter.get() - 1);
            }

            if can_maybe_skip_refresh
                && neighbor.piece.get().is_some()
                && neighbor.adjacent_pieces() == 2
            {
                self.frozen_map.clear(idx);
                self.frozen_map.set(Board::convert_coords_to_idx(
                    neighbor_pos.0,
                    neighbor_pos.1,
                    self.dim,
                ));
                can_maybe_skip_refresh = false;
                needs_frozen_refresh = false;
            }

            // The neighbor is now a potentially playable location.
            if prev_piece.is_none()
                && neighbor.adjacent_pieces() == 1
                && neighbor.piece.get().is_none()
            {
                let curr_head = NonZeroUsize::new(Board::convert_coords_to_idx(
                    neighbor_pos.0,
                    neighbor_pos.1,
                    self.dim,
                ));

                self.add_to_linked_list(neighbor, curr_head, &self.first_playable_tile);
            }
        }

        if needs_frozen_refresh {
            self.needs_frozen_refresher.set(true);
        }

        Ok(())
    }

    fn internal_remove_piece(&mut self, pos: Position) -> Result<Piece> {
        let idx = Board::convert_coords_to_idx(pos.0, pos.1, self.dim);
        if idx == 0 {
            return Err(HiveError::ZeroPositionIsUnplayable);
        }

        let Some(cell) = self.grid.get(idx) else {
            return Err(HiveError::PositionOutOfBounds);
        };

        if self.frozen_map.get(idx) && !self.has_stacked_pieces(pos) {
            return Err(HiveError::PositionIsFrozen);
        }

        let Some(piece) = cell.piece.take() else {
            return Err(HiveError::PositionIsEmpty);
        };

        let mut uncovered_piece = None;
        if let Some(idx) = cell.covered_piece.take() {
            debug_assert!(piece.is_beetle());

            let CoveredPiece { piece, next } = self.covered_pieces_two.remove(idx.get());
            uncovered_piece = Some(piece);
            cell.piece.set(Some(piece));
            cell.covered_piece.set(next);
        }

        let mut needs_frozen_refresh = true;
        let mut can_maybe_skip_refresh_based_on_neighbor = cell.adjacent_pieces() == 1;
        if uncovered_piece.is_some() {
            // If we're uncovering another piece, then we haven't changed the structure of the hive at all.
            needs_frozen_refresh = false;
            can_maybe_skip_refresh_based_on_neighbor = false;
        }

        // Fix up the linked list between pieces.
        let curr_head = NonZeroUsize::new(idx);
        let pred_head = cell.prev_piece.take();
        let succ_head = cell.next_piece.take();
        let list_head = self.get_ll_by_color(piece.color);

        self.remove_from_linked_list(curr_head, pred_head, succ_head, list_head);

        // Configure the tile based on the piece that we just uncovered.
        let list_to_join = if let Some(uncovered_piece) = uncovered_piece {
            self.get_ll_by_color(uncovered_piece.color)
        } else {
            &self.first_playable_tile
        };

        self.add_to_linked_list(cell, curr_head, list_to_join);

        for (_, neighbor_pos) in pos.neighbors(self.dim) {
            let neighbor = self
                .get(neighbor_pos)
                .expect("should be internally consistent");

            let adjacent_piece_counter = neighbor.adjacent_pieces.get(piece.color);
            adjacent_piece_counter.set(adjacent_piece_counter.get() - 1);
            if let Some(uncovered_color) = uncovered_piece.map(|p| p.color) {
                let uncovered_piece_counter = neighbor.adjacent_pieces.get(uncovered_color);

                uncovered_piece_counter.set(uncovered_piece_counter.get() + 1);
            }

            if can_maybe_skip_refresh_based_on_neighbor
                && neighbor.piece.get().is_some()
                && neighbor.adjacent_pieces() == 1
            {
                self.frozen_map.clear(idx);
                self.frozen_map.clear(Board::convert_coords_to_idx(
                    neighbor_pos.0,
                    neighbor_pos.1,
                    self.dim,
                ));

                // set this to false so that we can short circuit this on the next iteration of the loop.
                can_maybe_skip_refresh_based_on_neighbor = false;
                needs_frozen_refresh = false;
            }

            // The neighbor was a potentially playable location and now it isn't.
            if neighbor.adjacent_pieces() == 0 && neighbor.piece.get().is_none() {
                // debug_assert!(neighbor.piece.get().is_none());

                let curr_head = NonZeroUsize::new(Board::convert_coords_to_idx(
                    neighbor_pos.0,
                    neighbor_pos.1,
                    self.dim,
                ));

                let pred_head = neighbor.prev_piece.take();
                let succ_head = neighbor.next_piece.take();

                self.remove_from_linked_list(
                    curr_head,
                    pred_head,
                    succ_head,
                    &self.first_playable_tile,
                );
            }
        }

        if needs_frozen_refresh {
            self.needs_frozen_refresher.set(true);
        }

        Ok(piece)
    }

    fn is_frozen(&self, pos: Position) -> bool {
        let idx = Board::convert_coords_to_idx(pos.0, pos.1, self.dim);
        self.frozen_map.get(idx)
    }

    fn has_stacked_pieces(&self, pos: Position) -> bool {
        self[pos].covered_piece.get().is_some()
    }

    pub fn enumerate_playable_positions<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = Position> + 'a {
        let is_first_piece_for_second_player = self.pieces_on_board.get() == 1;

        let start = if self.pieces_on_board.get() == 0 {
            NonZeroUsize::new(Board::convert_coords_to_idx(
                self.dim / 2,
                self.dim / 2,
                self.dim,
            ))
        } else {
            self.first_playable_tile.get()
        };

        TilesIter {
            board: self,
            current: start,
        }
        .filter(move |pos| {
            let cell = self.get(*pos).expect("internally consistent");
            debug_assert!(cell.piece.get().is_none());

            match color {
                _ if is_first_piece_for_second_player => true,
                Color::Black if cell.adjacent_pieces.white().get() == 0 => true,
                Color::White if cell.adjacent_pieces.black().get() == 0 => true,
                _ => false,
            }
        })
    }

    pub fn enumerate_pieces(&self, color: Color) -> TilesIter {
        let starting_pos = match color {
            Color::Black => self.first_black_piece.get(),
            Color::White => self.first_white_piece.get(),
        };

        TilesIter {
            board: self,
            current: starting_pos,
        }
    }

    pub fn enumerate_all_pieces(&self) -> impl Iterator<Item = Position> + '_ {
        self.enumerate_pieces(Color::White)
            .chain(self.enumerate_pieces(Color::Black))
    }

    pub fn enumerate_free_pieces<'a>(
        &'a self,
        color: Color,
    ) -> impl Iterator<Item = (Piece, Position)> + 'a {
        let ll = self.get_ll_by_color(color);

        TilesIter {
            board: self,
            current: ll.get(),
        }
        .filter_map(move |pos| {
            let cell = &self[pos];
            let piece = cell.piece.get().expect("internally consistent");
            if !self.is_frozen(pos) {
                Some((piece, pos))
            } else if cell.covered_piece.get().is_some() {
                Some((piece, pos))
            } else {
                None
            }
        })
    }

    pub fn enumerate_all_pieces_on_tile<'a>(
        &'a self,
        pos: Position,
    ) -> impl Iterator<Item = Piece> + 'a {
        let cell = &self[pos];

        let mut curr_covered_idx = cell.covered_piece.get();
        let enumerate_covered = std::iter::from_fn(move || {
            if let Some(idx) = curr_covered_idx {
                let covered = self.covered_pieces_two[idx.get()];

                curr_covered_idx = covered.next;
                Some(covered.piece)
            } else {
                None
            }
        });

        cell.piece.get().into_iter().chain(enumerate_covered)
    }

    /// For an occupied position `pos`, this function will enumerate all the valid moves
    /// that the piece at that position can make given the rest of the board. Returns an error
    ///
    /// # Errors
    ///
    /// Will return an error if the provided position is out of bounds of the board, if there
    /// is no piece at the position, or if the piece is frozen in place by the pieces surrouding
    /// it.
    pub fn enumerate_valid_moves(&self, pos: Position) -> Result<PieceMovements> {
        let Some(cell) = self.get(pos) else {
            return Err(HiveError::PositionOutOfBounds);
        };

        let Some(piece) = cell.piece.get() else {
            return Err(HiveError::PositionIsEmpty);
        };

        if self.is_frozen(pos) {
            if cell.covered_piece.get().is_none() {
                return Err(HiveError::PositionIsFrozen);
            }
        }

        Ok(match piece.role {
            Insect::QueenBee => {
                PieceMovements::QueenBeeMove(movement::QueenBeeMovement::new(self, pos))
            }
            Insect::Beetle => PieceMovements::BeetleMove(movement::BeetleMovement::new(self, pos)),
            Insect::Grasshopper => {
                PieceMovements::GrasshopperMove(movement::GrasshopperMovement::new(self, pos))
            }
            Insect::Spider => PieceMovements::SpiderMove(movement::SpiderMovement::new(self, pos)),
            Insect::SoldierAnt => {
                PieceMovements::SoldierAntMove(movement::SoldierAntMovement::new(self, pos))
            }
        })
    }

    fn add_to_linked_list(
        &self,
        cell: &GridCell,
        curr_head: Option<NonZeroUsize>,
        list_head: &Cell<Option<NonZeroUsize>>,
    ) {
        // Put the item on the front of the LL,
        let prev_head = list_head.replace(curr_head);
        cell.next_piece.set(prev_head);

        if let Some(old_idx) = prev_head {
            let old_cell = self
                .grid
                .get(old_idx.get())
                .expect("should be internally consistent");

            debug_assert!(old_cell.prev_piece.get().is_none());
            old_cell.prev_piece.set(curr_head);
        }
    }

    fn debug_linked_list(&self, mut head: Option<NonZeroUsize>) {
        while let Some(item) = head {
            println!("{}", item);
            head = self.grid[item.get()].next_piece.get();
        }
    }

    fn remove_from_linked_list(
        &self,
        curr_head: Option<NonZeroUsize>,
        pred_head: Option<NonZeroUsize>,
        succ_head: Option<NonZeroUsize>,
        list_head: &Cell<Option<NonZeroUsize>>,
    ) {
        // The cell is empty, but it won't be after this method exits. We need to remove it from the list
        // of empty/playable cells.

        // Check if we're removing the front of the list.
        if list_head.get() == curr_head {
            debug_assert!(pred_head.is_none());
            list_head.set(succ_head);
        }

        if let Some(pred_head) = pred_head {
            let pred_cell = self
                .grid
                .get(pred_head.get())
                .expect("should be internally consistent");

            debug_assert!(pred_cell.next_piece.get() == curr_head);
            pred_cell.next_piece.set(succ_head);
        }

        if let Some(succ_head) = succ_head {
            let succ_cell = self
                .grid
                .get(succ_head.get())
                .expect("should be internally consistent");

            debug_assert!(succ_cell.prev_piece.get() == curr_head);
            succ_cell.prev_piece.set(pred_head);
        }
    }

    fn get_ll_by_color(&self, color: Color) -> &Cell<Option<NonZeroUsize>> {
        match color {
            Color::Black => &self.first_black_piece,
            Color::White => &self.first_white_piece,
        }
    }

    fn convert_coords_to_idx(x: u8, y: u8, dim: u8) -> usize {
        y as usize * dim as usize + x as usize
    }

    fn convert_idx_to_coords(idx: usize, dim: u8) -> Position {
        let y = idx / dim as usize;
        let x = idx % dim as usize;

        Position(x as u8, y as u8)
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::new(BOARD_SIZE)
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        if self.dim != other.dim {
            return false;
        }

        if self.pieces_on_board.get() != other.pieces_on_board.get() {
            return false;
        }

        for (lhs, rhs) in self.grid.iter().zip(other.grid.iter()) {
            if lhs.piece != rhs.piece {
                return false;
            }

            match (lhs.covered_piece.get(), rhs.covered_piece.get()) {
                (None, None) => {}
                (Some(lhs_idx), Some(rhs_idx)) => {
                    let mut lhs_it = Some(lhs_idx);
                    let mut rhs_it = Some(rhs_idx);
                    while let Some(lhs_val) = lhs_it {
                        let Some(rhs_val) = rhs_it else {
                            return false;
                        };

                        let lhs_covered_piece = &self.covered_pieces_two[lhs_val.get()];
                        let rhs_covered_piece = &self.covered_pieces_two[rhs_val.get()];
                        if lhs_covered_piece.piece != rhs_covered_piece.piece {
                            return false;
                        }

                        lhs_it = lhs_covered_piece.next;
                        rhs_it = rhs_covered_piece.next;
                    }

                    if rhs_it.is_some() {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }
}

impl Eq for Board {}

impl Hash for Board {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for x in self.grid.iter() {
            x.piece.get().hash(state);
        }
    }
}

#[derive(Debug)]
pub struct BoardInconsistencyStats {
    pub number_of_hives: usize,
    pub inconsistent_hexes: usize,
    pub inconsistent_white_pieces: Option<(usize, usize)>,
    pub inconsistent_black_pieces: Option<(usize, usize)>,
}

pub struct TilesIter<'a> {
    board: &'a Board,
    current: Option<NonZeroUsize>,
}

impl Iterator for TilesIter<'_> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        let cell = self
            .board
            .grid
            .get(current.get())
            .expect("should be internally consistent");

        let pos = Board::convert_idx_to_coords(current.get(), self.board.dim);

        self.current = cell.next_piece.get();

        Some(pos)
    }
}

pub enum PieceMovements<'a> {
    BeetleMove(movement::BeetleMovement<'a>),
    QueenBeeMove(movement::QueenBeeMovement<'a>),
    SoldierAntMove(movement::SoldierAntMovement<'a>),
    GrasshopperMove(movement::GrasshopperMovement<'a>),
    SpiderMove(movement::SpiderMovement<'a>),
}

impl<'a> Iterator for PieceMovements<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PieceMovements::BeetleMove(movement) => movement.next(),
            PieceMovements::QueenBeeMove(movement) => movement.next(),
            PieceMovements::SoldierAntMove(movement) => movement.next(),
            PieceMovements::GrasshopperMove(movement) => movement.next(),
            PieceMovements::SpiderMove(movement) => movement.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fmt::Write};

    use crate::piece::Insect;

    use super::*;

    fn assert_no_error(f: impl FnOnce() -> Result<()>) {
        assert_eq!(f(), Ok(()));
    }

    #[test]
    fn first_piece_locations() {
        let mut b = Board::new(30);
        let Some(x) = b.enumerate_playable_positions(Color::Black).next() else {
            panic!("should have found a position");
        };

        b.place_piece_in_empty_cell(
            Piece {
                role: Insect::Grasshopper,
                color: Color::Black,
            },
            x,
        )
        .expect("success");

        let v = b
            .enumerate_playable_positions(Color::White)
            .collect::<HashSet<_>>();

        for p in [
            Position(14, 15),
            Position(14, 16),
            Position(15, 16),
            Position(16, 15),
            Position(16, 14),
            Position(15, 14),
        ] {
            assert!(v.contains(&p));
        }
    }

    #[test]
    fn second_piece_locations() {
        assert_no_error(|| {
            let mut b = Board::new(30);
            let Some(position) = b.enumerate_playable_positions(Color::White).next() else {
                unreachable!("should have found a position");
            };

            b.place_piece_in_empty_cell(
                Piece {
                    role: Insect::Spider,
                    color: Color::White,
                },
                position,
            )?;

            b.validate_board().unwrap();

            let Some(position) = b.enumerate_playable_positions(Color::Black).next() else {
                unreachable!("should have found a position");
            };

            b.place_piece_in_empty_cell(
                Piece {
                    role: Insect::QueenBee,
                    color: Color::Black,
                },
                position,
            )?;

            b.validate_board().unwrap();

            Ok(())
        });
    }

    // #[test]
    // fn test_tarjan_paths() {
    //     assert_no_error(|| {
    //         let mut b = Board::default();

    //         // First black piece
    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::Beetle,
    //                 color: Color::Black,
    //             },
    //             Position(2, 2),
    //         )?;

    //         // First white piece.
    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::Spider,
    //                 color: Color::White,
    //             },
    //             Position(2, 3),
    //         )?;

    //         // Remaining black pieces adjacent to the first black piece.
    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::QueenBee,
    //                 color: Color::Black,
    //             },
    //             Position(1, 2),
    //         )?;

    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::SoldierAnt,
    //                 color: Color::Black,
    //             },
    //             Position(2, 1),
    //         )?;

    //         // Remaining white pieces.
    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::Beetle,
    //                 color: Color::White,
    //             },
    //             Position(2, 4),
    //         )?;

    //         b.place_piece_in_empty_cell(
    //             Piece {
    //                 role: Insect::Beetle,
    //                 color: Color::White,
    //             },
    //             Position(1, 4),
    //         )?;

    //         let mut iter = tarjan::BoardBridgeIter::new(Position(2, 2), &b);
    //         assert!(matches!(
    //             iter.next(),
    //             Some((Position(2, 2), Position(2, 3)))
    //         ));
    //         assert_eq!(iter.next(), None);

    //         let mut frozen_black_pieces = b.enumerate_frozen_pieces(Color::Black);
    //         assert_eq!(frozen_black_pieces.next(), Some(Position(2, 2)));
    //         assert_eq!(frozen_black_pieces.next(), None);

    //         let mut frozen_black_pieces = b.enumerate_frozen_pieces(Color::White);
    //         assert_eq!(frozen_black_pieces.next(), Some(Position(2, 3)));
    //         assert_eq!(frozen_black_pieces.next(), None);

    //         Ok(())
    //     });
    // }

    #[test]
    fn test_dense_format() {
        let b = r#"
        . . . . . . 
         . . b s s .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let b = Board::from_string_repr(&b, true).expect("success");

        let mut s = String::new();
        write!(&mut s, "{}", b.dense_format()).expect("success");
        println!("{}", s);

        let b2 = Board::from_dense_repr(&s, b.dim).expect("deserialized");
        println!("{}", b2);

        assert_eq!(b, b2);
    }
}
