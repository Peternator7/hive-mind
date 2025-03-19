use core::f32;
use std::time::Instant;

use transposition::{CachedBound, TranspositionTable};

use crate::{
    game::{Game, GameWinner},
    movement::Move,
    piece::Color,
    Result,
};

mod transposition;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum MoveStrengthHeuristic {
    WinningMove,
    BestMoveFromPriorIteration,

    MoveOnTopOfEnemyQueen,
    MoveAdjacentToEnemyQueen,

    MoveAdjacentToEnemyPiece,

    NonDescriptMove,
    PiecePlacement,

    MoveOnTopOfOwnQueen,
    MoveOffOfEnemyQueen,

    LosingMove,
}

pub struct AgentConfig {
    max_depth: usize,
}

pub struct Agent {
    explored_positions: TranspositionTable,
    scratch: AgentScratch,
    config: AgentConfig,
}

#[derive(Default, Debug, PartialEq, PartialOrd)]
struct Stats {
    depth: usize,
    scores_calculated: usize,
    beta_cutoffs: usize,
    hypothetical_beta_cutoffs: usize,
    nws_attempted: usize,
    nws_requires_research: usize,
    first_move_is_best: usize,
    first_move_is_not_best: usize,
    transposition_table_size: usize,
    transposition_table_hashes_seen: usize,
    transposition_table_hits: usize,
    transposition_caused_low_fail: usize,
    transposition_caused_high_fail: usize,
    transposition_table_exact_match: usize,
    time_take: f32,
    best_move: Option<(f32, Move)>,
}

#[derive(Debug)]
struct MovePlus {
    strength: MoveStrengthHeuristic,
    mv: Move,
}

#[derive(Default, Debug)]
struct AgentScratch {
    moves_vectors: Vec<Vec<MovePlus>>,
}

impl Agent {
    pub fn new(target_depth: usize) -> Self {
        Self {
            scratch: Default::default(),
            explored_positions: Default::default(),
            config: AgentConfig {
                max_depth: target_depth,
            },
        }
    }

    pub async fn determine_best_move(&mut self, game: &Game) -> crate::Result<Move> {
        let mut game = game.clone();
        let mut best_move = None;
        let mut last_stats = Stats::default();
        for x in 0..self.config.max_depth {
            let mut stats = Stats::default();
            let now = Instant::now();

            best_move = self
                .determine_best_move_at_depth(&mut game, &mut stats, x)
                .await?;

            stats.time_take = (Instant::now() - now).as_secs_f32();
            let (size, hashes) = self.explored_positions.size_stats();
            stats.transposition_table_size = size;
            stats.transposition_table_hashes_seen = hashes;
            stats.depth = x;
            stats.best_move = best_move;
            last_stats = stats;
        }

        dbg!(last_stats);
        Ok(best_move.map(|mv| mv.1).unwrap_or(Move::Pass))
    }

    async fn determine_best_move_at_depth(
        &mut self,
        game: &mut Game,
        stats: &mut Stats,
        depth: usize,
    ) -> Result<Option<(f32, Move)>> {
        let mut cached_best_move = None;
        if let Some(entry) = self.explored_positions.get(game) {
            stats.transposition_table_hits += 1;
            // We're only interested in positions that have explored at least as deep
            // as this iteration will require.
            if entry.depth >= depth {
                match entry.bound {
                    CachedBound::Exact => {
                        stats.transposition_table_exact_match += 1;
                        return Ok(Some((entry.score, entry.best_move.unwrap())));
                    }
                    CachedBound::Invalid => unreachable!("not properly initialized"),
                    _ => {}
                }
            }

            cached_best_move = entry.best_move;
        }

        let mut all_moves = self.scratch.moves_vectors.pop().unwrap_or_default();
        all_moves.clear();

        self.calculate_all_potential_moves(game, &mut all_moves, cached_best_move)?;
        all_moves.retain(|mv| !game.is_repeat_move(mv.mv, 64));
        all_moves.sort_unstable_by(|lhs, rhs| lhs.strength.cmp(&rhs.strength));

        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;

        if depth == 0 {
            let mut best_move: Option<(f32, Move)> = None;

            for mv in all_moves.drain(..) {
                game.make_move(mv.mv)?;

                let mut score = game.calculate_score();
                if game.to_play() == Color::Black {
                    score = -score;
                }

                game.unmake_move()?;

                stats.scores_calculated += 1;
                best_move = match best_move {
                    Some((curr_best, _)) if score > curr_best => Some((score, mv.mv)),
                    None => Some((score, mv.mv)),
                    _ => best_move,
                };
            }

            self.scratch.moves_vectors.push(all_moves);
            return Ok(best_move);
        }

        let mut best_move = None;
        let mut full_window_search = true;

        for mv_plus in all_moves.drain(..) {
            game.make_move(mv_plus.mv)?;

            let score = if full_window_search {
                Box::pin(self.pvs(game, stats, depth - 1, -beta, -alpha))
                    .await?
                    .map(|sc| -sc)
            } else {
                stats.nws_attempted += 1;
                let nws = Box::pin(self.pvs(game, stats, depth - 1, -alpha - f32::EPSILON, -alpha))
                    .await?
                    .map(|sc| -sc);

                match nws {
                    Some(nws) if alpha < nws && nws < beta => {
                        // Redo search with full window.
                        stats.nws_requires_research += 1;
                        Box::pin(self.pvs(game, stats, depth - 1, -beta, -alpha))
                            .await?
                            .map(|sc| -sc)
                    }
                    _ => nws,
                }
            };

            game.unmake_move()?;

            let Some(score) = score else {
                continue;
            };

            let mut can_skip_alpha_beta_check = true;
            match best_move {
                Some((curr_best, _)) if score > curr_best => {
                    best_move = Some((score, mv_plus.mv));
                    can_skip_alpha_beta_check = false;
                }
                None => {
                    best_move = Some((score, mv_plus.mv));
                    can_skip_alpha_beta_check = false;
                }
                _ => {}
            }

            if can_skip_alpha_beta_check {
                continue;
            }

            if score > alpha {
                alpha = score;
                full_window_search = false;
            }

            if alpha >= beta {
                stats.beta_cutoffs += 1;
                break;
            }
        }

        if let Some((score, mv)) = best_move {
            let cache = self.explored_positions.get_mut_or_default(game);
            cache.depth = depth;
            cache.score = score;
            cache.best_move = Some(mv);
            cache.transposition_key = game.transposition_key();

            // We found a candidate PV.
            cache.bound = CachedBound::Exact;
        }

        if cached_best_move == best_move.map(|x| x.1) {
            stats.first_move_is_best += 1;
        } else {
            stats.first_move_is_not_best += 1;
        }

        self.scratch.moves_vectors.push(all_moves);
        Ok(best_move)
    }

    async fn pvs(
        &mut self,
        game: &mut Game,
        stats: &mut Stats,
        depth: usize,
        mut alpha: f32,
        beta: f32,
    ) -> Result<Option<f32>> {
        let initial_alpha = alpha;
        let mut cached_best_move = None;

        // Check if the game is over.
        match game.is_game_is_over() {
            None => {}
            Some(GameWinner::Tie) => return Ok(Some(0.0)),
            Some(GameWinner::Winner(winner)) if winner == game.to_play() => return Ok(Some(100.0)),
            Some(GameWinner::Winner(..)) => return Ok(Some(-100.0)),
        }

        if let Some(entry) = self.explored_positions.get(game) {
            stats.transposition_table_hits += 1;
            // We're only interested in positions that have explored at least as deep
            // as this iteration will require.
            if entry.depth >= depth {
                match entry.bound {
                    CachedBound::LowFail => {
                        if entry.score < alpha {
                            stats.transposition_caused_low_fail += 1;
                            return Ok(Some(entry.score));
                        }
                    }
                    CachedBound::HighFail => {
                        if entry.score >= beta {
                            stats.transposition_caused_high_fail += 1;
                            return Ok(Some(entry.score));
                        }
                    }
                    CachedBound::Exact => {
                        stats.transposition_table_exact_match += 1;
                        return Ok(Some(entry.score));
                    }
                    CachedBound::Invalid => unreachable!("not properly initialized"),
                }
            }

            cached_best_move = entry.best_move;
        }

        let mut all_moves = self.scratch.moves_vectors.pop().unwrap_or_default();
        all_moves.clear();

        self.calculate_all_potential_moves(game, &mut all_moves, cached_best_move)?;
        all_moves.sort_unstable_by(|lhs, rhs| lhs.strength.cmp(&rhs.strength));

        if depth == 0 {
            let output = self.calculate_best_score_of_leaf_node(
                game,
                stats,
                beta,
                all_moves.drain(..).map(|m| m.mv),
            );
            self.scratch.moves_vectors.push(all_moves);
            return output;
        }

        let mut best_move = None;
        let mut full_window_search = true;

        for mv_plus in all_moves.drain(..) {
            game.make_move(mv_plus.mv)?;

            let score = if full_window_search {
                Box::pin(self.pvs(game, stats, depth - 1, -beta, -alpha))
                    .await?
                    .map(|sc| -sc)
            } else {
                stats.nws_attempted += 1;
                let nws = Box::pin(self.pvs(game, stats, depth - 1, -alpha - f32::EPSILON, -alpha))
                    .await?
                    .map(|sc| -sc);

                match nws {
                    Some(nws) if alpha < nws && nws < beta => {
                        // Redo search with full window.
                        stats.nws_requires_research += 1;
                        Box::pin(self.pvs(game, stats, depth - 1, -beta, -alpha))
                            .await?
                            .map(|sc| -sc)
                    }
                    _ => nws,
                }
            };

            game.unmake_move()?;

            let Some(score) = score else {
                continue;
            };

            let mut can_skip_alpha_beta_check = true;
            match best_move {
                Some((curr_best, _)) if score > curr_best => {
                    best_move = Some((score, mv_plus.mv));
                    can_skip_alpha_beta_check = false;
                }
                None => {
                    best_move = Some((score, mv_plus.mv));
                    can_skip_alpha_beta_check = false;
                }
                _ => {}
            }

            if can_skip_alpha_beta_check {
                continue;
            }

            if score > alpha {
                alpha = score;
                full_window_search = false;
            }

            if alpha >= beta {
                stats.beta_cutoffs += 1;
                break;
            }
        }

        // return the borrowed vec.
        self.scratch.moves_vectors.push(all_moves);

        if let Some((score, mv)) = best_move {
            let cache = self.explored_positions.get_mut_or_default(game);
            cache.depth = depth;
            cache.score = score;
            cache.best_move = Some(mv);
            cache.transposition_key = game.transposition_key();

            if score <= initial_alpha {
                // Low Fail
                cache.bound = CachedBound::LowFail;
            } else if alpha >= beta {
                // High Fail
                cache.bound = CachedBound::HighFail;
            } else {
                // We found a candidate PV.
                cache.bound = CachedBound::Exact;
            }

            if cached_best_move == Some(mv) {
                stats.first_move_is_best += 1;
            } else {
                stats.first_move_is_not_best += 1;
            }

            Ok(Some(score))
        } else {
            Ok(None)
        }
    }

    fn calculate_best_score_of_leaf_node(
        &self,
        game: &mut Game,
        stats: &mut Stats,
        beta: f32,
        moves: impl Iterator<Item = Move>,
    ) -> Result<Option<f32>> {
        let mut best_move: Option<f32> = None;

        game.set_frozenness_checkpoint();
        for mv in moves {
            game.make_move(mv)?;

            let mut score = game.calculate_score();
            if game.to_play() == Color::Black {
                score = -score;
            }

            game.unmake_move()?;

            stats.scores_calculated += 1;
            best_move = match best_move {
                Some(curr_best) if score > curr_best => Some(score),
                None => Some(score),
                _ => best_move,
            };

            if score >= beta {
                break;
            }
        }

        game.clear_frozenness_checkpoint();
        Ok(best_move)
    }

    fn calculate_all_potential_moves(
        &self,
        game: &Game,
        all_moves: &mut Vec<MovePlus>,
        mut cached_best_move: Option<Move>,
    ) -> Result<()> {
        let to_play = game.to_play();
        let turn = game.turn();

        let queen_in_hand = game.hand(to_play).has_queen();
        let can_move = !queen_in_hand;
        let can_play_queen = turn > 1;
        let must_play_queen = turn == 4 && queen_in_hand;

        for piece in game.enumerate_playable_pieces_from_hand(to_play) {
            if !can_play_queen && piece.is_queen() {
                continue;
            }

            if must_play_queen && !piece.is_queen() {
                continue;
            }

            for position in game.enumerate_playable_positions(to_play) {
                let mv = Move::PlacePiece { piece, position };
                let mut strength = self.estimate_move_strength(game, mv);
                if let Some(best_mv) = cached_best_move {
                    if best_mv == mv {
                        strength = MoveStrengthHeuristic::BestMoveFromPriorIteration;
                        cached_best_move = None;
                    }
                }

                all_moves.push(MovePlus { strength, mv })
            }
        }

        if !can_move {
            return Ok(());
        }

        for (piece, from) in game.enumerate_free_pieces(to_play) {
            for to in game.enumerate_valid_moves(from)? {
                let mv = Move::MovePiece { piece, from, to };
                let mut strength = self.estimate_move_strength(game, mv);
                if let Some(best_mv) = cached_best_move {
                    if best_mv == mv {
                        strength = MoveStrengthHeuristic::BestMoveFromPriorIteration;
                        cached_best_move = None;
                    }
                }

                all_moves.push(MovePlus { strength, mv });
            }
        }

        Ok(())
    }

    fn estimate_move_strength(&self, game: &Game, mv: Move) -> MoveStrengthHeuristic {
        match mv {
            Move::MovePiece { piece, to, from } => {
                if let Some(queen_pos) = game.queen_pos(piece.color.opposing()) {
                    if to == queen_pos {
                        return MoveStrengthHeuristic::MoveOnTopOfEnemyQueen;
                    }

                    if from == queen_pos {
                        return MoveStrengthHeuristic::MoveOffOfEnemyQueen;
                    }

                    if to.is_adjacent(&queen_pos) {
                        if game.board()[queen_pos].adjacent_pieces() == 5
                            && game.board()[to].piece.get().is_none()
                        {
                            return MoveStrengthHeuristic::WinningMove;
                        }

                        return MoveStrengthHeuristic::MoveAdjacentToEnemyQueen;
                    }
                }

                if let Some(queen_pos) = game.queen_pos(piece.color) {
                    if to == queen_pos {
                        return MoveStrengthHeuristic::MoveOnTopOfOwnQueen;
                    }

                    if to.is_adjacent(&queen_pos)
                        && game.board()[queen_pos].adjacent_pieces() == 5
                        && game.board()[to].piece.get().is_none()
                    {
                        return MoveStrengthHeuristic::LosingMove;
                    }
                }

                if game.board()[to]
                    .adjacent_pieces
                    .get(piece.color.opposing())
                    .get()
                    > 0
                {
                    return MoveStrengthHeuristic::MoveAdjacentToEnemyPiece;
                }

                MoveStrengthHeuristic::NonDescriptMove
            }
            Move::PlacePiece { .. } => MoveStrengthHeuristic::PiecePlacement,
            Move::Pass => unreachable!(),
        }
    }
}
