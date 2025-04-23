use fastbloom::BloomFilter;
use hive_engine::game::Game;

pub struct SeenPositions {
    generations: Vec<fastbloom::BloomFilter>,
    size: usize,
    curr_gen_size: usize,
}

impl SeenPositions {
    pub fn new(generations: usize) -> Self {
        Self {
            size: 0,
            curr_gen_size: 0,
            generations: (0..generations)
                .map(|_| BloomFilter::with_false_pos(0.001).expected_items(500_000))
                .collect(),
        }
    }

    pub fn advance_generation(&mut self) {
        println!("Seen Size: {}", self.curr_gen_size);
        self.curr_gen_size = 0;
        self.size += 1;
        let l = self.generations.len();
        if self.size >= l {
            self.generations[self.size % l].clear();
        }
    }

    pub fn is_unseen_position(&mut self, game: &Game) -> bool {
        let l = self.generations.len();
        for i in 0..l {
            if i > self.size {
                break;
            }

            let idx = (self.size - i) % l;
            if self.generations[idx].contains(&game.transposition_key()) {
                if i > 0 {
                    self.generations[self.size % l].insert(&game.transposition_key());
                    self.curr_gen_size += 1;
                }

                return false;
            }
        }

        self.curr_gen_size += 1;
        self.generations[self.size % l].insert(&game.transposition_key());
        true
    }
}
