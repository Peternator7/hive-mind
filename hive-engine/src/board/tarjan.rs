use crate::position::{NeighborsIter, Position};

use super::{cache::ActuationPointIterStateCache, Board};

#[derive(Debug, Copy, Clone)]
pub struct BoardBridgeIterState {
    disc: usize,
    scc: usize,
}

#[derive(Debug, Clone)]
pub(super) struct OnStackState {
    curr: Position,
    came_from: Position,
    children: usize,
    neighbors: NeighborsIter,
    low: usize,
    disc: usize,
}

pub struct ActuationPointIter<'a> {
    board: &'a Board,
    idx: usize,
    disc_time: Vec<(Position, usize)>,
    on_stack: Vec<OnStackState>,
}

impl<'a> ActuationPointIter<'a> {
    pub fn new(start_pos: Position, board: &'a Board) -> Self {
        debug_assert!(board
            .get(start_pos)
            .expect("internally consistent")
            .piece
            .get()
            .is_some());

        let reusable = board
            .cache
            .borrow_mut()
            .articulation_point_cache
            .pop()
            .unwrap_or_default();
        let mut on_stack = reusable.on_stack;
        on_stack.clear();
        on_stack.push(OnStackState {
            curr: start_pos,
            came_from: start_pos,
            children: 0,
            neighbors: start_pos.neighbors(board.dim),
            low: 0,
            disc: 0,
        });

        let mut disc_time: Vec<(Position, usize)> = reusable.state;
        disc_time.clear();
        disc_time.push((start_pos, 0));

        ActuationPointIter {
            idx: 0,
            board,
            on_stack,
            disc_time,
        }
    }
}

impl<'a> Drop for ActuationPointIter<'a> {
    fn drop(&mut self) {
        self.board
            .cache
            .borrow_mut()
            .articulation_point_cache
            .push(ActuationPointIterStateCache {
                on_stack: std::mem::take(&mut self.on_stack),
                state: std::mem::take(&mut self.disc_time),
            });
    }
}

impl<'a> Iterator for ActuationPointIter<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let mut returned_from = None;

        'outer_loop: while let Some(top_of_stack) = self.on_stack.last_mut() {
            let &mut OnStackState {
                curr,
                came_from,
                disc: curr_disc,
                ref mut children,
                ref mut neighbors,
                ref mut low,
            } = top_of_stack;
            if let Some(neighbor_low) = returned_from {
                // If our neighbor found was able to join a lower rank scc than the one we're currently a part
                // of, we should join it to. Transitively, this means that our neighbor found a loop back to one
                // of our parents.
                *low = (*low).min(neighbor_low);

                // There are 2 different scenarios we consider:
                // 1. If the neighbors "group" is the same as our discovery index, that means
                //    the neighbor found a path back to this node, but nothing "better"
                //    That means this node is critical to the graph *if* we aren't the root node
                //    of the search.
                // 2. If we are looking at the root node AND the root node we've explored
                //    more than one child node, that means that these 2 children are not
                //    connected and therefore removing this node would disconnect two halves
                //    of the graph.
                if neighbor_low >= curr_disc && curr != came_from {
                    return Some(curr);
                }

                if curr == came_from && *children > 1 {
                    return Some(curr);
                }

                returned_from = None;
            }

            // Visit each of our neighboring nodes.
            while let Some((_, neighbor)) = neighbors.next() {
                // Don't revisit the parent node.
                if neighbor == came_from {
                    continue;
                }

                // We don't look at empty spaces for determining bridges.
                if self
                    .board
                    .get(neighbor)
                    .expect("internally consistent")
                    .piece
                    .get()
                    .is_none()
                {
                    continue;
                }

                match self
                    .disc_time
                    .iter()
                    .rev()
                    .find(|(key, _)| *key == neighbor)
                {
                    // The neighbor has already been visited.
                    Some(&(_, neighbor_state)) => {
                        // Check to see if someone found a faster way to get to our neighbor than to us.
                        *low = (*low).min(neighbor_state);
                    }
                    // The neighbor is unvisited.
                    None => {
                        self.idx += 1;

                        *children += 1;
                        self.disc_time.push((neighbor, self.idx));
                        self.on_stack
                            //.push((neighbor, curr, 0, neighbor.neighbors(self.board.dim)));
                            .push(OnStackState {
                                curr: neighbor,
                                came_from: curr,
                                children: 0,
                                neighbors: neighbor.neighbors(self.board.dim),
                                low: self.idx,
                                disc: self.idx,
                            });
                        // This is the "recursive step"
                        continue 'outer_loop;
                    }
                }
            }

            // We have no more neighbors to visit.
            // Pop this node off the stack, set `returned_from` so the node we came from knows
            // that we backtracked.
            let low = *low;
            self.on_stack.pop();
            returned_from = Some(low);
        }

        None
    }
}
