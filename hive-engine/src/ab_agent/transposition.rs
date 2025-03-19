use std::collections::{HashMap, HashSet};

use crate::{game::Game, movement::Move, piece::Color};

#[derive(Default)]
pub struct TranspositionTable {
    map: HashMap<(Color, u64), TableEntry>,
    //map: HashMap<(Color, String), TableEntry>,
    transposition_keys_seen: HashSet<(Color, u64)>,
}

#[derive(Default)]
pub struct TableEntry {
    pub depth: usize,
    pub score: f32,
    pub bound: CachedBound,
    pub best_move: Option<Move>,
    pub transposition_key: u64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CachedBound {
    /// The move we have cached for this position was less good than a move we have already seen.
    /// This is equivalent to best_move <= alpha.
    LowFail,
    /// The best move in this board is probably "too" good.
    /// best_move >= beta.
    HighFail,
    /// We computed the exact score of this move. That means it's a candidate to be a part of the
    /// the Principal Variation
    Exact,

    Invalid,
}

impl Default for CachedBound {
    fn default() -> Self {
        Self::Invalid
    }
}

impl TranspositionTable {
    pub fn get(&mut self, game: &Game) -> Option<&TableEntry> {
        let to_play = game.to_play();
        // let mut scratch = self.buf.take();
        // scratch.clear();
        // write!(&mut scratch, "{}", game.board().dense_format()).unwrap();

        // let key = (to_play, scratch);
        // let output = self.map.get(&key);
        // if let Some(entry) = output {
        //     debug_assert_eq!(entry.transposition_key, game.transposition_key());
        // }

        // self.buf.set(key.1);
        // output
        self.map.get(&(to_play, game.transposition_key()))
    }

    pub fn get_mut_or_default(&mut self, game: &Game) -> &mut TableEntry {
        let to_play = game.to_play();
        let key = (to_play, game.transposition_key());
        self.map.entry(key).or_default()

        // let mut scratch = self.buf.take();
        // scratch.clear();
        // write!(&mut scratch, "{}", game.board().dense_format()).unwrap();

        // let key = (to_play, scratch);
        // if self.map.contains_key(&key) {
        //     // The entry api, while great would require us to reallocate the string
        //     // on each check.
        //     let output = self.map.get_mut(&key).unwrap();
        //     // self.buf.set(key.1);
        //     output
        // } else {
        //     self.transposition_keys_seen
        //         .insert((to_play, game.transposition_key()));
        //     self.map.entry(key).or_insert_with(Default::default)
        // }
    }

    pub fn size_stats(&self) -> (usize, usize) {
        (self.map.len(), self.transposition_keys_seen.len())
    }
}
