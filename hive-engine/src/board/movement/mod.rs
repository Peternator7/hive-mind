use std::{
    collections::{hash_map::Entry, HashMap},
    num::NonZeroUsize,
};

use crate::position::{Direction, DirectionMap, NeighborsIter, Position};

use super::Board;

pub struct BeetleMovement<'a>(BeetleQueenMovement<'a>);

impl<'a> BeetleMovement<'a> {
    pub(super) fn new(board: &'a Board, position: Position) -> Self {
        BeetleMovement(BeetleQueenMovement::new(board, position, true))
    }
}

impl<'a> Iterator for BeetleMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct QueenBeeMovement<'a>(BeetleQueenMovement<'a>);

impl<'a> QueenBeeMovement<'a> {
    pub(super) fn new(board: &'a Board, position: Position) -> Self {
        QueenBeeMovement(BeetleQueenMovement::new(board, position, false))
    }
}

impl<'a> Iterator for QueenBeeMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

struct BeetleQueenMovement<'a> {
    board: &'a Board,
    starting_pos: Position,
    candidates: NeighborsIter,
    can_aid_in_traversal: DirectionMap<bool>,
    can_move_onto_other_tiles: bool,
}

impl<'a> BeetleQueenMovement<'a> {
    fn new(board: &'a Board, position: Position, can_move_onto_other_tiles: bool) -> Self {
        let candidates = position.neighbors(board.dim);

        BeetleQueenMovement {
            board,
            candidates,
            starting_pos: position,
            can_aid_in_traversal: Default::default(),
            can_move_onto_other_tiles,
        }
    }
}

impl<'a> Iterator for BeetleQueenMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((direction, candidate)) = self.candidates.next() {
            let cell = self.board.get(candidate).expect("internally consistent");

            if cell.piece.get().is_some() {
                // Make a note that this tile can't be used for traversal by another tile.
                self.can_aid_in_traversal.set(direction, false);
                if self.can_move_onto_other_tiles {
                    return Some(candidate);
                }

                continue;
            }

            // The neighbor tile is empty.
            debug_assert!(cell.adjacent_pieces() > 0);
            if cell.adjacent_pieces() == 1 {
                // This tile is only adjacent to the current piece. That means we can't
                // move to it or we'd disconnect from the rest of the hive.
                //
                // We still record that the tile was empty to allow neighboring tiles
                // to benefit from this information.
                self.can_aid_in_traversal.set(direction, true);
                continue;
            }

            // By getting here, we've confirmed that the square is empty and
            if can_slide_from_src_in_direction(
                &self.board,
                self.starting_pos,
                direction,
                &mut self.can_aid_in_traversal,
            ) {
                return Some(candidate);
            }
        }

        None
    }
}

pub struct SoldierAntMovement<'a>(AntSpiderMovement<'a>);

impl<'a> SoldierAntMovement<'a> {
    pub(super) fn new(board: &'a Board, position: Position) -> Self {
        SoldierAntMovement(AntSpiderMovement::new(board, position, None))
    }
}

impl<'a> Iterator for SoldierAntMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct SpiderMovement<'a>(AntSpiderMovement<'a>);

impl<'a> SpiderMovement<'a> {
    pub(super) fn new(board: &'a Board, position: Position) -> Self {
        SpiderMovement(AntSpiderMovement::new(
            board,
            position,
            Some(NonZeroUsize::new(3).unwrap()),
        ))
    }
}

impl<'a> Iterator for SpiderMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

struct AntSpiderMovement<'a> {
    board: &'a Board,
    stack: Vec<(Position, NeighborsIter)>,
    node_state: HashMap<Position, NodeState>,
    target_depth: Option<NonZeroUsize>,
}

#[derive(Debug, Default, Clone)]
struct NodeState {
    on_stack: bool,
    can_skip_node: bool,
    visited_depths: usize,
    neighbor_cache: DirectionMap<bool>,
}

impl<'a> AntSpiderMovement<'a> {
    pub fn new(
        board: &'a Board,
        starting_pos: Position,
        target_depth: Option<NonZeroUsize>,
    ) -> Self {
        let mut node_state: HashMap<Position, NodeState> = HashMap::new();
        node_state.entry(starting_pos).or_default().on_stack = true;

        AntSpiderMovement {
            board,
            node_state,
            stack: vec![(starting_pos, starting_pos.neighbors(board.dim))],
            target_depth,
        }
    }

    pub fn get_current_path(&self) -> &[(Position, NeighborsIter)] {
        &self.stack
    }
}

impl<'a> Iterator for AntSpiderMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let starting_pos = self.stack.first().map(|(pos, _)| *pos).unwrap_or_default();
        let mut stack_len = self.stack.len();
        'outer_loop: while let Some(&mut (current_position, ref mut neighbors_iter)) =
            self.stack.last_mut()
        {
            while let Some((direction, neighbor)) = neighbors_iter.next() {
                // First check to see if we've already visisted the node.
                let mut entry = self.node_state.entry(neighbor);
                if let Entry::Occupied(ref mut neighbor_state) = entry {
                    let neighbor_state = neighbor_state.get_mut();
                    if neighbor_state.can_skip_node {
                        // Some other check has determined we don't need to explore this node.
                        continue;
                    }

                    if self.target_depth.is_none() {
                        // If we don't have a target depth, don't bother re-checking this node.
                        // If we do have a target depth, we can potentially explore it again
                        // to see if we've found different solutions.
                        continue;
                    }

                    // We are allowed to revisit nodes, but we never backtrack
                    // so don't explore the node if it's part of the current path.
                    if neighbor_state.on_stack {
                        // This is a backedge.
                        continue;
                    }

                    if (neighbor_state.visited_depths & (1 << stack_len)) != 0 {
                        continue;
                    }
                }

                let neighbor_cell = self.board.get(neighbor).expect("internally consistent");

                if neighbor_cell.piece.get().is_some() {
                    entry.or_default().can_skip_node = true;
                    continue;
                }

                let adjacent_pieces = neighbor_cell.adjacent_pieces();
                if adjacent_pieces == 0 {
                    // All empty cells adjacent to the hive are on the "playable" squares list.
                    // If the cell isn't in that list, then we definitely can't move there.
                    entry.or_default().can_skip_node = true;
                    continue;
                }

                if adjacent_pieces == 1 && neighbor.is_adjacent(&starting_pos) {
                    // If the cell is adjacent to the starting position and it's only touching 1 piece,
                    // that piece is this piece and we would be disconnecting the hive by moving to it.
                    entry.or_default().can_skip_node = true;
                    continue;
                }

                let traversability_cache = &mut self
                    .node_state
                    .entry(current_position)
                    .or_default()
                    .neighbor_cache;

                if !can_slide_from_src_in_direction(
                    &self.board,
                    current_position,
                    direction,
                    traversability_cache,
                ) {
                    continue;
                }

                // We can move from current_position to the new position.
                // Either add it to the stack or return it if we've hit the target depth.
                if let Some(target_depth) = self.target_depth {
                    if target_depth.get() == stack_len {
                        return Some(neighbor);
                    }
                }

                let node_state = self.node_state.entry(neighbor).or_default();
                node_state.on_stack = true;

                // Mark that we found this node at this depth if we're tracking that.
                if self.target_depth.is_some() {
                    node_state.visited_depths |= 1 << stack_len;
                }

                self.stack
                    .push((neighbor, neighbor.neighbors(self.board.dim)));

                stack_len = self.stack.len();
                continue 'outer_loop;
            }

            // Pop the current item off the stack.
            self.stack.pop();
            stack_len = self.stack.len();

            // Return the position if we're not tracking depth.
            // We also don't return the starting_position.
            if stack_len > 0 && self.target_depth.is_none() {
                return Some(current_position);
            }
        }

        None
    }
}

pub struct GrasshopperMovement<'a> {
    board: &'a Board,
    neighbors: NeighborsIter,
}

impl<'a> GrasshopperMovement<'a> {
    pub fn new(board: &'a Board, position: Position) -> Self {
        GrasshopperMovement {
            board,
            neighbors: position.neighbors(board.dim),
        }
    }
}

impl<'a> Iterator for GrasshopperMovement<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((direction, candidate)) = self.neighbors.next() {
            let mut jumped_over_something = false;
            let mut destination = Some(candidate);
            while let Some(landing_spot) = destination {
                let has_piece = self
                    .board
                    .get(landing_spot)
                    .expect("internally consistent")
                    .piece
                    .get()
                    .is_some();

                if has_piece {
                    jumped_over_something = true;
                    // Move another tile in the same direction and see if the
                    // grasshopper can land there.
                    destination = landing_spot.neighbor(direction, self.board.dim);
                    continue;
                } else if jumped_over_something {
                    // The grasshopper must jump over at least one piece.
                    return Some(landing_spot);
                } else {
                    break;
                }
            }

            // There are two scenarios where we get here:
            // 1. the immediate neighbor is empty so the square is empty, but
            //    the grasshopper didn't jump over anything to get there (likely scenario)
            // 2. we reached the edge of the board so there's not a valid move. (unlikely)
            // In either case, we just check the next direction.
        }

        None
    }
}

/// Check to see if we can move from `starting_pos` in `direction`. This function assumes that
/// the caller has already checked that the tile in `direction` square is not occupied. It's
/// responsibility is to check whether the tiles adjacent to `direction` are empty so that the tile
/// can "slide" from starting_pos to the new position.
fn can_slide_from_src_in_direction(
    board: &Board,
    starting_pos: Position,
    direction: Direction,
    can_aid_in_traversal: &mut DirectionMap<bool>,
) -> bool {
    // By getting here, we've confirmed that the square is empty and
    match can_aid_in_traversal.get(direction) {
        // The implication here is subtle. If an adjacent tile has marked the current
        // tile as helpful before we processed it, that's because the adjacent tile is also
        // empty. Therefore we can return immediately because we have two empty adjacent
        // tiles.
        Some(true) => return true,
        Some(false) => unreachable!("checked at top of function."),
        None => {
            let neighbor_directions = [
                direction.rotate_counter_clockwise(),
                direction.rotate_clockwise(),
            ];

            for neighbor_direction in neighbor_directions {
                match can_aid_in_traversal.get(neighbor_direction) {
                    Some(true) => {
                        // If the neighbor tile is traversable, and this tile is empty, then this tile is also traversable.
                        can_aid_in_traversal.set(direction, true);
                        return true;
                    }
                    Some(false) => {} // If the neighbor tile is not traversable
                    None => {
                        let mut neighbor_cell_empty = true;
                        if let Some(neighbor_pos) =
                            starting_pos.neighbor(neighbor_direction, board.dim)
                        {
                            let neighbor_cell =
                                board.get(neighbor_pos).expect("internally consistent");

                            if neighbor_cell.piece.get().is_some() {
                                // If the neighbor tile is occupied, then the tile is not traversable.
                                neighbor_cell_empty = false;
                                can_aid_in_traversal.set(neighbor_direction, false);
                            }
                        }

                        if neighbor_cell_empty {
                            // We're at the edge of the board. The piece isn't valid for placement, but it would be correct
                            // to slide a piece through it I guess 🤷‍♂️
                            can_aid_in_traversal.set(direction, true);
                            // By extension of our own cell being empty, the neighbor cell is also
                            // now traversable.
                            can_aid_in_traversal.set(neighbor_direction, true);
                            return true;
                        }
                    }
                }
            }

            // If we get here, it means that both of the neighboring tiles are occupied. We'll just loop again
            // and try the next candidate.
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::{board::PieceMovements, piece::Color};

    use super::*;

    #[test]
    fn test_ant() {
        let b = r#"
        . . . . . . 
         . . b s s .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let mut b = Board::from_string_repr(&b, true).expect("success");
        let start = Position(3, 2);

        let PieceMovements::SoldierAntMove(it) = b.enumerate_valid_moves(start).expect("testcase")
        else {
            panic!("expected ant movement");
        };

        let mut expected_values = [
            Position(3, 3),
            Position(2, 4),
            Position(1, 5),
            Position(0, 5),
            Position(0, 4),
            Position(0, 3),
            Position(0, 2),
            Position(1, 1),
            Position(2, 0),
            Position(3, 0),
            Position(4, 0),
            Position(4, 1),
        ]
        .into_iter()
        .collect::<HashSet<_>>();

        for end in dbg!(it.collect::<Vec<_>>()) {
            assert!(expected_values.remove(&end));
            b.move_piece(start, end).expect("testcase");
            b.validate_board().expect("testcase");
            b.move_piece(end, start).expect("testcase");
            b.validate_board().expect("testcase");
        }

        assert!(expected_values.is_empty());
    }

    #[test]
    fn test_queen() {
        let b = r#"
        . . . . . . 
         . . b s G .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let b = Board::from_string_repr(&b, true).expect("success");
        let pos = Position(2, 1);

        let PieceMovements::QueenBeeMove(it) = b.enumerate_valid_moves(pos).expect("testcase")
        else {
            panic!("expected queen movement");
        };

        let mut expected_values = [Position(3, 0), Position(1, 1)]
            .into_iter()
            .collect::<HashSet<_>>();

        for valid_movement in it {
            assert!(expected_values.remove(&valid_movement));
        }

        assert!(expected_values.is_empty());
    }

    #[test]
    fn test_beetle() {
        let b = r#"
        . . . . . . 
         . . b s G .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let b = Board::from_string_repr(&b, true).expect("success");
        let pos = Position(1, 2);

        let PieceMovements::BeetleMove(it) = b.enumerate_valid_moves(pos).expect("testcase") else {
            panic!("expected beetle movement");
        };

        let it = dbg!(it.collect::<Vec<_>>());

        let mut expected_values = [
            Position(0, 3),
            Position(1, 3),
            Position(2, 1),
            Position(1, 1),
        ]
        .into_iter()
        .collect::<HashSet<_>>();

        for valid_movement in it {
            assert!(expected_values.remove(&valid_movement));
        }

        assert!(expected_values.is_empty());
    }

    #[test]
    fn test_beetle_stacking() {
        let b = r#"
        . . . . . . 
         . . b s G .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let mut b = Board::from_string_repr(&b, true).expect("success");
        let start = Position(1, 2);

        // Test moving the beetle onto empty spaces and stacking it onto pieces of the same
        // and opposite colors.
        for end in [Position(1, 3), Position(2, 1), Position(0, 3)] {
            b.move_piece(start, end).expect("testcase");
            b.validate_board().expect("testcase");

            println!("Move succeeded");

            b.move_piece(end, start).expect("testcase");
            println!("{}", b);
            b.validate_board().expect("testcase");
        }
    }

    #[test]
    fn test_beetle_stacking_2() {
        let b = r#"
        . . . . . . . .
         . . . . . . . .
          . . . . q . . .
           . . B b a . . .
            . . Q . . . . .
             . . . . . . . .
              . . . . . . . .
      "#;

        let b = Board::from_string_repr(&b, true).expect("success");

        let free_pieces = dbg!(b.enumerate_free_pieces(Color::Black).collect::<Vec<_>>());
        assert_eq!(2, free_pieces.len());
    }

    #[test]
    fn test_beetle_stacking_3() {
        let b = r#"
        . . . . . . . .
         . . . . . . . .
          . . . A s . . .
           . . . b Q . . .
            . . q . . . . .
             . . . . . . . .
              . . . . . . . .
      "#;

        let b = Board::from_string_repr(&b, true).expect("success");

        // b.recalculate_frozen_cells();
        let free_pieces = dbg!(b.enumerate_free_pieces(Color::White).collect::<Vec<_>>());
        assert_eq!(2, free_pieces.len());
    }

    #[test]
    fn test_beetle_stacking_4() {
        let b = r#"
        . . . . . . . .
         . . . . . s . .
          . . . . a q . .
           . S Q B b . . .
            . . a . . . . .
             . . . . . . . .
              . . . . . . . .
      "#;

        let b = Board::from_string_repr(&b, true).expect("success");
        b.recalculate_frozen_cells();

        let free_pieces = dbg!(b.enumerate_free_pieces(Color::Black).collect::<Vec<_>>());
        assert_eq!(1, free_pieces.len());
    }

    #[test]
    fn test_spider() {
        let b = r#"
        . . . . . . 
         . . b s G .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let mut b = Board::from_string_repr(&b, true).expect("success");
        let start = Position(3, 1);

        let PieceMovements::SpiderMove(it) = b.enumerate_valid_moves(start).expect("testcase")
        else {
            panic!("expected spider movement");
        };

        let it = dbg!(it.collect::<Vec<_>>());

        let mut expected_values = [Position(3, 3), Position(1, 1)]
            .into_iter()
            .collect::<HashSet<_>>();

        for end in it {
            assert!(expected_values.remove(&end));
            b.move_piece(start, end).expect("testcase");
            b.validate_board().expect("testcase");
            b.move_piece(end, start).expect("testcase");
            b.validate_board().expect("testcase");
        }

        assert!(expected_values.is_empty());
    }

    #[test]
    fn test_grasshopper() {
        let b = r#"
        . . . . . .
         . . b s G .
          . Q . q . .
           . S A . . .
            . . . . . .
             . . . . . .
        "#;

        let b = Board::from_string_repr(&b, true).expect("success");
        let pos = Position(1, 4);

        let PieceMovements::GrasshopperMove(it) = b.enumerate_valid_moves(pos).expect("testcase")
        else {
            panic!("expected grasshopper movement");
        };

        let it = dbg!(it.collect::<Vec<_>>());

        let mut expected_values = [Position(4, 1), Position(1, 1)]
            .into_iter()
            .collect::<HashSet<_>>();

        for valid_movement in it {
            assert!(expected_values.remove(&valid_movement));
        }

        assert!(expected_values.is_empty());
    }
}
