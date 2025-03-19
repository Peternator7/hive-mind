use core::f32;
use std::{
    fmt::Display,
    fs::File,
    io::{BufWriter, Write},
    sync::atomic::{AtomicUsize, Ordering},
};

static SCORES_CALCULATED: AtomicUsize = AtomicUsize::new(0);
static BRANCHES_PRUNED: AtomicUsize = AtomicUsize::new(0);
const PRINT_EVERY_X_SCORES: usize = 1_000_000;
const SEARCH_DEPTH: usize = 1;

use crate::{
    board::{frozen_map::FrozenMap, Board},
    error::HiveError,
    hand::Hand,
    piece::{Color, Insect, Piece},
    position::Position,
};

pub struct Agent<'a> {
    pub color: Color,
    pub turn: usize,
    pub hand: &'a mut Hand,
    pub validate: bool,
}

struct Scratch {
    pos_vectors: Vec<Vec<Position>>,
    pos_piece_vectors: Vec<Vec<(Piece, Position)>>,
    frozen_map: FrozenMap,
    buf_writer: Option<BufWriter<File>>,
    move_stack: Vec<Move>,
}

#[derive(Debug, Copy, Clone)]
pub enum Move {
    PlacePiece {
        piece: Piece,
        position: Position,
    },
    MovePiece {
        piece: Piece,
        from: Position,
        to: Position,
    },
    Pass,
    Tie,
}

impl Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Move::PlacePiece { piece, position } => write!(f, "{} # {}", piece, position),
            Move::MovePiece { piece, from, to } => write!(f, "{} # {} -> {}", piece, from, to),
            Move::Pass => write!(f, "Pass"),
            Move::Tie => write!(f, "Tie"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentError {
    HiveError(HiveError),
    Context {
        color: Color,
        turn: usize,
        movement: Move,
        backtracking: bool,
        inner: Box<AgentError>,
    },
}

impl Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::HiveError(hive_error) => write!(f, "{}", hive_error),
            AgentError::Context {
                color,
                turn,
                movement,
                backtracking,
                inner,
            } => write!(
                f,
                "[{}][{}][{}][{}] -> {}",
                turn, color, movement, backtracking, inner
            ),
        }
    }
}

impl std::error::Error for AgentError {}

#[derive(Debug)]
struct PredictedResponse {
    score: f32,
    best_move: Move,
}

impl<'a> Agent<'a> {
    fn print_stats(&self) {
        println!(
            "Scores Computed: {}, Branches Pruned: {}, AP's Avoid: {}, AP's Required: {}",
            SCORES_CALCULATED.load(Ordering::Relaxed),
            BRANCHES_PRUNED.load(Ordering::Relaxed),
            crate::board::RECALCULATE_SKIPPED.load(Ordering::Relaxed),
            crate::board::RECALCULATE_REQUIRED.load(Ordering::Relaxed),
        )
    }

    pub fn determine_best_move(
        &mut self,
        board: &mut Board,
        opponent: &mut Agent,
    ) -> std::result::Result<Move, AgentError> {
        let frozen_map = board.frozenness();

        // let output_writer = Some(BufWriter::new(
        //     File::create("samples/output.txt").expect("success"),
        // ));

        let mut scratch = Scratch {
            frozen_map,
            move_stack: Default::default(),
            pos_piece_vectors: Default::default(),
            pos_vectors: Default::default(),
            buf_writer: None,
        };

        let output = self.determine_board_score(
            board,
            opponent,
            f32::NEG_INFINITY,
            f32::INFINITY,
            SEARCH_DEPTH,
            &mut scratch,
        )?;

        if let Some(buf_writer) = scratch.buf_writer.as_mut() {
            buf_writer.flush().expect("flush failed");
        }

        // dbg!(&output);

        self.print_stats();

        Ok(output.best_move)
    }

    fn determine_board_score(
        &mut self,
        board: &mut Board,
        opponent: &mut Agent,
        mut alpha: f32,
        mut beta: f32,
        depth: usize,
        scratch: &mut Scratch,
    ) -> std::result::Result<PredictedResponse, AgentError> {
        let mut scratch1 = scratch.pos_vectors.pop().unwrap_or_default();
        let mut scratch3 = scratch.pos_piece_vectors.pop().unwrap_or_default();

        let hand_contains_queen = self.hand.has_queen();
        let can_move = !hand_contains_queen;
        let must_play_queen = hand_contains_queen && self.turn == 4;
        let can_play_queen = self.turn > 1;

        // Clear the scratch vectors.
        scratch1.clear();
        scratch3.clear();

        let mut tried_something = false;
        let mut can_exit_early = false;

        let mut curr_best_response = PredictedResponse {
            score: 0.0,
            best_move: Move::Tie,
        };

        if self.color == Color::White {
            curr_best_response.score = f32::NEG_INFINITY;
        } else {
            curr_best_response.score = f32::INFINITY;
        }

        if depth == 0 {
            board.clone_frozenness(&mut scratch.frozen_map);
        }

        scratch1.extend(board.enumerate_playable_positions(self.color));

        for p in scratch1.drain(..) {
            for insect in Insect::iter() {
                if must_play_queen && insect != Insect::QueenBee {
                    continue;
                }

                if !can_play_queen && insect == Insect::QueenBee {
                    continue;
                }

                let Some(insect) = self.hand.pop_tile(insect) else {
                    continue;
                };

                tried_something = true;
                let tile = Piece {
                    role: insect,
                    color: self.color,
                };

                let mv = Move::PlacePiece {
                    piece: tile,
                    position: p,
                };

                scratch.move_stack.push(mv);
                board
                    .place_piece_in_empty_cell(tile, p)
                    .add_agent_context(self.color, self.turn, false, mv)?;

                let output =
                    self.do_recursive_step(board, opponent, alpha, beta, depth, mv, scratch)?;

                can_exit_early = self.do_alpha_beta_check(
                    &mut curr_best_response,
                    mv,
                    output.score,
                    &mut alpha,
                    &mut beta,
                );

                // UNDO the placement
                self.hand.push_tile(insect);
                let removed_piece = if depth > 0 {
                    board
                        .remove_piece(p)
                        .add_agent_context(self.color, self.turn, true, mv)?
                } else {
                    // To optimize the number of leaf nodes that we process, we cache the
                    // frozen map and then restore it to avoid needing to do the graph traversal.
                    board
                        .remove_piece_and_restore_frozen_map(p, &scratch.frozen_map)
                        .add_agent_context(self.color, self.turn, true, mv)?
                };

                scratch.move_stack.pop();
                assert_eq!(removed_piece, tile);

                if self.validate {
                    board
                        .validate_board()
                        .add_agent_context(self.color, self.turn, true, mv)?;
                }

                if can_exit_early {
                    // It's important that we undo what we did before we break out of the loop.
                    break;
                }
            }
        }

        if can_exit_early {
            scratch.pos_vectors.push(scratch1);
            scratch.pos_piece_vectors.push(scratch3);
            return Ok(curr_best_response);
        }

        if can_move {
            // If we didn't break out of the above loop, do this one.
            scratch3.extend(board.enumerate_free_pieces(self.color));

            // Someone on github suggested sorting the pieces so that SoldierAnt moves are tried last.
            scratch3.sort_unstable_by_key(|p| p.0.role);

            for (piece, src) in scratch3.drain(..) {
                match board
                    .enumerate_valid_moves(src)
                    .expect("failed to enumerate moves")
                {
                    crate::board::PieceMovements::BeetleMove(it) => scratch1.extend(it),
                    crate::board::PieceMovements::QueenBeeMove(it) => scratch1.extend(it),
                    crate::board::PieceMovements::SoldierAntMove(it) => scratch1.extend(it),
                    crate::board::PieceMovements::GrasshopperMove(it) => scratch1.extend(it),
                    crate::board::PieceMovements::SpiderMove(it) => scratch1.extend(it),
                }

                assert_eq!(board[src].piece.get(), Some(piece));
                for dst in scratch1.drain(..) {
                    let mv = Move::MovePiece {
                        piece: piece,
                        from: src,
                        to: dst,
                    };

                    scratch.move_stack.push(mv);
                    board
                        .move_piece(src, dst)
                        .add_agent_context(self.color, self.turn, false, mv)?;

                    if self.validate {
                        board
                            .validate_board()
                            .add_agent_context(self.color, self.turn, false, mv)?;
                    }

                    tried_something = true;

                    let output =
                        self.do_recursive_step(board, opponent, alpha, beta, depth, mv, scratch)?;

                    can_exit_early = self.do_alpha_beta_check(
                        &mut curr_best_response,
                        mv,
                        output.score,
                        &mut alpha,
                        &mut beta,
                    );

                    // undo the move. Every move should be valid to do in reverse.
                    scratch.move_stack.pop();
                    if depth > 0 {
                        board
                            .move_piece(dst, src)
                            .add_agent_context(self.color, self.turn, true, mv)?;
                    } else {
                        board
                            .move_piece_and_restore_frozen_map(dst, src, &scratch.frozen_map)
                            .add_agent_context(self.color, self.turn, true, mv)?;
                    }

                    if self.validate {
                        board
                            .validate_board()
                            .add_agent_context(self.color, self.turn, true, mv)?;
                    }

                    if can_exit_early {
                        // Same as above. We need to undo the move before we break out of the loop.
                        break;
                    }
                }
            }
        }

        if !tried_something {
            // Pass the move back to our opponent without doing anything.
            if depth > 0 {
                let output = opponent.determine_board_score(
                    board,
                    &mut Agent {
                        color: self.color,
                        turn: self.turn + 1,
                        hand: &mut self.hand,
                        validate: self.validate,
                    },
                    alpha,
                    beta,
                    depth - 1,
                    scratch,
                )?;
                curr_best_response.score = output.score;
                if let Move::Pass = output.best_move {
                    // If the other side can't make any moves then this game is a tie.
                    curr_best_response.best_move = Move::Tie;
                } else {
                    curr_best_response.best_move = Move::Pass;
                }
            } else {
                // println!("what happened: {}", board.print_board());
                curr_best_response.score = 0.0;
            }
        }

        scratch.pos_vectors.push(scratch1);
        scratch.pos_piece_vectors.push(scratch3);
        Ok(curr_best_response)
    }

    fn do_recursive_step(
        &mut self,
        board: &mut Board,
        opponent: &mut Agent,
        alpha: f32,
        beta: f32,
        depth: usize,
        mv: Move,
        scratch: &mut Scratch,
    ) -> std::result::Result<PredictedResponse, AgentError> {
        if self.validate {
            board
                .validate_board()
                .add_agent_context(self.color, self.turn, false, mv)?;
        }

        let output;
        if depth > 0 {
            output = opponent
                .determine_board_score(
                    board,
                    &mut Agent {
                        color: self.color,
                        turn: self.turn + 1,
                        hand: self.hand,
                        validate: self.validate,
                    },
                    alpha,
                    beta,
                    depth - 1,
                    scratch,
                )
                .add_agent_context(self.color, self.turn, false, mv)?;

            return Ok(output);
        } else {
            println!("{:?}", &scratch.move_stack);
            let counter = SCORES_CALCULATED.fetch_add(1, Ordering::Relaxed);
            if counter % PRINT_EVERY_X_SCORES == 0 {
                self.print_stats();
            }

            return Ok(PredictedResponse {
                score: board.score(),
                best_move: mv,
            });
        }
    }

    fn do_alpha_beta_check(
        &mut self,
        curr_best_response: &mut PredictedResponse,
        candidate_move: Move,
        candidate_score: f32,
        alpha: &mut f32,
        beta: &mut f32,
    ) -> bool {
        if self.color == Color::White {
            if candidate_score > curr_best_response.score {
                curr_best_response.score = candidate_score;
                curr_best_response.best_move = candidate_move;

                *alpha = alpha.max(curr_best_response.score);
            }
        } else {
            if candidate_score < curr_best_response.score {
                curr_best_response.score = candidate_score;
                curr_best_response.best_move = candidate_move;

                *beta = beta.min(curr_best_response.score);
            }
        }

        if beta <= alpha {
            BRANCHES_PRUNED.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }
}

trait AgentContext {
    type Output;

    fn add_agent_context(
        self,
        color: Color,
        turn: usize,
        backtracking: bool,
        mv: Move,
    ) -> Self::Output;
}

impl<T> AgentContext for std::result::Result<T, HiveError> {
    type Output = std::result::Result<T, AgentError>;

    fn add_agent_context(
        self,
        color: Color,
        turn: usize,
        backtracking: bool,
        mv: Move,
    ) -> Self::Output {
        match self {
            Ok(output) => Ok(output),
            Err(e) => Err(AgentError::Context {
                color,
                turn,
                movement: mv,
                backtracking,
                inner: Box::new(AgentError::HiveError(e)),
            }),
        }
    }
}

impl<T> AgentContext for std::result::Result<T, AgentError> {
    type Output = std::result::Result<T, AgentError>;

    fn add_agent_context(
        self,
        color: Color,
        turn: usize,
        backtracking: bool,
        mv: Move,
    ) -> Self::Output {
        match self {
            Ok(output) => Ok(output),
            Err(e) => Err(AgentError::Context {
                color,
                turn,
                backtracking,
                movement: mv,
                inner: Box::new(e),
            }),
        }
    }
}
