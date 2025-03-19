
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Insect {
    Grasshopper,
    QueenBee,
    Beetle,
    Spider,
    SoldierAnt,
}

impl Insect {
    pub fn iter() -> impl Iterator<Item = Insect> {
        const INSECTS: [Insect; 5] = [
            Insect::Grasshopper,
            Insect::QueenBee,
            Insect::Beetle,
            Insect::Spider,
            Insect::SoldierAnt,
        ];

        INSECTS.iter().copied()
    }
}