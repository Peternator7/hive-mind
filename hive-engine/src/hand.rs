use crate::piece::Insect;

#[derive(Debug, Clone)]
pub struct Hand {
    queen_bee: usize,
    soldier_ants: usize,
    grasshoppers: usize,
    beetles: usize,
    spiders: usize,
}

impl Hand {
    pub fn new() -> Self {
        Self {
            queen_bee: 1,
            soldier_ants: 3,
            grasshoppers: 3,
            beetles: 2,
            spiders: 2,
        }
    }

    pub fn has_queen(&self) -> bool {
        self.has_insect(Insect::QueenBee)
    }

    pub fn has_insect(&self, insect: Insect) -> bool {
        match insect {
            Insect::Grasshopper => self.grasshoppers > 0,
            Insect::QueenBee => self.queen_bee > 0,
            Insect::Beetle => self.beetles > 0,
            Insect::Spider => self.spiders > 0,
            Insect::SoldierAnt => self.soldier_ants > 0,
        }
    }

    pub fn pop_tile(&mut self, insect: Insect) -> Option<Insect> {
        let counter = self.get_counter(insect);

        if *counter > 0 {
            *counter -= 1;
            Some(insect)
        } else {
            None
        }
    }

    pub fn push_tile(&mut self, insect: Insect) {
        *self.get_counter(insect) += 1;
    }

    fn get_counter(&mut self, insect: Insect) -> &mut usize {
        match insect {
            Insect::Grasshopper => &mut self.grasshoppers,
            Insect::QueenBee => &mut self.queen_bee,
            Insect::Beetle => &mut self.beetles,
            Insect::Spider => &mut self.spiders,
            Insect::SoldierAnt => &mut self.soldier_ants,
        }
    }
}


impl Default for Hand {
    fn default() -> Self {
        Self::new()
    }
}