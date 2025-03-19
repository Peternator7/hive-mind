use std::{cell::Cell, vec};

const BITS_PER_IDX: usize = 8 * std::mem::size_of::<usize>();

#[derive(Debug)]
pub struct FrozenMap {
    items: Box<[Cell<usize>]>,
}

impl FrozenMap {
    pub fn new(size: usize) -> Self {
        let mut vec_size = size / BITS_PER_IDX;
        if size % BITS_PER_IDX != 0 {
            vec_size += 1;
        }

        Self {
            items: vec![Default::default(); vec_size].into_boxed_slice(),
        }
    }

    pub fn get(&self, idx: usize) -> bool {
        let Some(item) = self.items.get(idx / BITS_PER_IDX) else {
            return false;
        };

        item.get() & Self::find_bit(idx) > 0
    }

    pub fn set(&self, idx: usize) {
        let Some(item) = self.items.get(idx / BITS_PER_IDX) else {
            return;
        };

        item.set(item.get() | Self::find_bit(idx))
    }

    pub fn clear(&self, idx: usize) {
        let Some(item) = self.items.get(idx / BITS_PER_IDX) else {
            return;
        };

        item.set(item.get() & !Self::find_bit(idx))
    }

    pub fn clear_all(&self) {
        for item in &self.items {
            item.set(0);
        }
    }

    fn find_bit(idx: usize) -> usize {
        1 << idx % BITS_PER_IDX
    }
}

impl Clone for FrozenMap {
    fn clone_from(&mut self, source: &Self) {
        self.items.clone_from_slice(&source.items);
    }

    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrozenMap;

    #[test]
    fn test_set() {
        const SIZE: usize = 128;

        let map = FrozenMap::new(SIZE);
        assert_eq!(2, map.items.len());
        for target in 0..SIZE {
            map.set(target);
            for test in 0..SIZE {
                assert_eq!(
                    map.get(test),
                    test == target,
                    "failed at target: {}, test: {}",
                    target,
                    test
                );
            }

            map.clear(target);
        }
    }
}
