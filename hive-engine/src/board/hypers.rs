use crate::piece::{Color, ColorMap};

use super::Board;

pub const POINTS_PER_PIECE: f32 = 1.0;
pub const POINTS_LOST_PER_FROZEN_PIECE: f32 = -2.0;
pub const POINTS_LOST_PER_OPPONENT_PIECE_TOUCHING_QUEEN: f32 = -4.0;
pub const POINTS_LOST_PER_TEAM_PIECE_TOUCHING_QUEEN: f32 = -2.0;
pub const POINTS_FOR_LOSING: f32 = -100.0;

#[derive(Debug, Clone)]
pub struct ScoreConfiguration {
    points_per_piece: f32,
    standard_piece_scoring: PerPieceScoreConfiguration,
    queen_piece_scoring: PerPieceScoreConfiguration,
    additional_queen_scoring: QueenAdditionalScoringConfiguration,
}

#[derive(Debug, Clone)]
pub struct PerPieceScoreConfiguration {
    cost_per_frozen_piece: f32,
    cost_per_covered_piece: f32,
}

#[derive(Debug, Clone)]
pub struct QueenAdditionalScoringConfiguration {
    cost_per_adjacent_friendly_piece: f32,
    cost_per_adjacent_opposing_piece: f32,
}

pub static INTERNAL_SCORE_CONFIGURATION: ScoreConfiguration = ScoreConfiguration {
    points_per_piece: 2.0,
    standard_piece_scoring: PerPieceScoreConfiguration {
        cost_per_frozen_piece: -2.0,
        cost_per_covered_piece: -2.0,
    },
    queen_piece_scoring: PerPieceScoreConfiguration {
        cost_per_frozen_piece: -3.0,
        cost_per_covered_piece: -4.0,
    },
    additional_queen_scoring: QueenAdditionalScoringConfiguration {
        cost_per_adjacent_friendly_piece: -1.5,
        cost_per_adjacent_opposing_piece: -2.0,
    },
};

pub enum Score {
    GameOver(GameOver),
    Score(f32),
}

pub enum GameOver {
    Winner(Color),
    Tie,
}

impl Board {
    pub fn score(&self) -> f32 {
        let config = &INTERNAL_SCORE_CONFIGURATION;

        let mut queen_surrounded_map = ColorMap::<bool>::default();
        let mut score_map = ColorMap::<f32>::default();

        for pos in self.enumerate_all_pieces() {
            let cell = &self[pos];
            let mut top_piece = true;
            for piece in self.enumerate_all_pieces_on_tile(pos) {
                *score_map.get_mut(piece.color) += POINTS_PER_PIECE;
                let scoring = if piece.is_queen() {
                    if cell.adjacent_pieces() == 6 {
                        queen_surrounded_map.set(piece.color, true);
                    }
    
                    let team_pieces = cell.adjacent_pieces.get(piece.color).get();
                    let opponent_pieces = cell.adjacent_pieces.get(piece.color.opposing()).get();
    
                    *score_map.get_mut(piece.color) += (team_pieces as f32) * config.additional_queen_scoring.cost_per_adjacent_friendly_piece;
                    *score_map.get_mut(piece.color) += (opponent_pieces as f32) * config.additional_queen_scoring.cost_per_adjacent_opposing_piece;

                    &config.queen_piece_scoring
                } else {
                    &config.standard_piece_scoring
                };

                if !top_piece {
                    *score_map.get_mut(piece.color) += scoring.cost_per_covered_piece;
                    continue;
                }

                if self.is_frozen(pos) {
                    if !piece.is_beetle() || cell.covered_piece.get().is_none() {
                        *score_map.get_mut(piece.color) += scoring.cost_per_frozen_piece;
                    }
                }

                top_piece = false;
            }
        }

        score_map.white() - score_map.black()
    }
}
