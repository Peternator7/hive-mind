use crate::piece::Insect;

#[derive(Debug, Clone)]
pub struct Hand {
    queen_bee: (usize, usize),
    soldier_ants: (usize, usize),
    grasshoppers: (usize, usize),
    beetles: (usize, usize),
    spiders: (usize, usize),
}

impl Hand {
    pub fn new() -> Self {
        Self {
            queen_bee: (1, 1),
            soldier_ants: (3, 3),
            grasshoppers: (3, 3),
            beetles: (2, 2),
            spiders: (2, 2),
        }
    }

    pub fn has_queen(&self) -> bool {
        self.has_insect(Insect::QueenBee)
    }

    pub fn has_insect(&self, insect: Insect) -> bool {
        match insect {
            Insect::Grasshopper => self.grasshoppers.0 > 0,
            Insect::QueenBee => self.queen_bee.0 > 0,
            Insect::Beetle => self.beetles.0 > 0,
            Insect::Spider => self.spiders.0 > 0,
            Insect::SoldierAnt => self.soldier_ants.0 > 0,
        }
    }

    pub fn next_insect_id(&self, insect: Insect) -> Option<usize> {
        let (curr, max) = match insect {
            Insect::Grasshopper => self.grasshoppers,
            Insect::QueenBee => self.queen_bee,
            Insect::Beetle => self.beetles,
            Insect::Spider => self.spiders,
            Insect::SoldierAnt => self.soldier_ants,
        };

        if curr == 0 {
            None
        } else {
            Some(max - curr)
        }
    }

    pub fn pop_tile(&mut self, insect: Insect) -> Option<(Insect, usize)> {
        let (counter, max) = self.get_counter_mut(insect);

        if *counter > 0 {
            *counter -= 1;
            let id = max - *counter - 1;
            Some((insect, id))
        } else {
            None
        }
    }

    pub fn push_tile(&mut self, insect: Insect) {
        *self.get_counter_mut(insect).0 += 1;
    }

    fn get_counter_mut(&mut self, insect: Insect) -> (&mut usize, usize) {
        match insect {
            Insect::Grasshopper => (&mut self.grasshoppers.0, self.grasshoppers.1),
            Insect::QueenBee => (&mut self.queen_bee.0, self.queen_bee.1),
            Insect::Beetle => (&mut self.beetles.0, self.beetles.1),
            Insect::Spider => (&mut self.spiders.0, self.spiders.1),
            Insect::SoldierAnt => (&mut self.soldier_ants.0, self.soldier_ants.1),
        }
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self::new()
    }
}
