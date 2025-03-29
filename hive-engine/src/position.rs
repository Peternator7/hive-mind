use std::fmt;

use crate::error::HiveError;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Position(pub u8, pub u8);

impl Position {
    pub fn is_adjacent(&self, neighbor: &Position) -> bool {
        let Position(x1, y1) = *self;
        let Position(x2, y2) = *neighbor;

        let x_delta = x1 as isize - x2 as isize;
        let y_delta = y1 as isize - y2 as isize;

        match (x_delta, y_delta) {
            (-1, 0) | (1, 0) | (0, -1) | (0, 1) | (1, -1) | (-1, 1) => true,
            _ => false,
        }
    }

    pub fn neighbors(&self, dim: u8) -> NeighborsIter {
        NeighborsIter {
            center: *self,
            state: Some(Direction::TopLeft),
            dim: dim - 1,
        }
    }

    pub fn neighbor(&self, direction: Direction, dim: u8) -> Option<Position> {
        let Position(x, y) = *self;
        let dim = dim - 1;
        match direction {
            Direction::TopRight => {
                if x == 0 {
                    return None;
                }

                if y == dim {
                    return None;
                }

                return Some(Position(x - 1, y + 1));
            }
            Direction::Right => {
                if y == dim {
                    return None;
                }

                return Some(Position(x, y + 1));
            }
            Direction::BottomRight => {
                if x == dim {
                    return None;
                }

                return Some(Position(x + 1, y));
            }
            Direction::BottomLeft => {
                if x == dim {
                    return None;
                }

                if y == 0 {
                    return None;
                }

                return Some(Position(x + 1, y - 1));
            }
            Direction::Left => {
                if y == 0 {
                    return None;
                }

                return Some(Position(x, y - 1));
            }
            Direction::TopLeft => {
                if x == 0 {
                    return None;
                }

                return Some(Position(x - 1, y));
            }
        }
    }

    pub fn to_cube_coords(self) -> (isize, isize, isize) {
        let Position(x, y) = self;

        (x as isize, y as isize, 0 - x as isize - y as isize)
    }

    fn from_cube_coords(x: isize, y: isize, z: isize, dim: u8) -> crate::Result<Self> {
        let dim = dim as isize;
        let _ = z;
        if x < 0 || x >= dim {
            return Err(HiveError::RotationOutOfBounds);
        }

        if y < 0 || y >= dim {
            return Err(HiveError::PositionOutOfBounds);
        }

        Ok(Self(x as u8, y as u8))
    }

    pub fn rotate_clockwise_around_center(
        self,
        center: Position,
        dim: u8,
    ) -> crate::Result<Position> {
        let (x, y, z) = self.to_cube_coords();
        let (c_x, c_y, c_z) = center.to_cube_coords();

        // Re-center after subtracting the offset.
        let (x, y, z) = (x - c_x, y - c_y, z - c_z);

        // Rotate + add center back in.
        let (x, y, z) = (-y + c_x, -z + c_y, -x + c_z);

        Self::from_cube_coords(x, y, z, dim)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Direction {
    TopRight,
    Right,
    BottomRight,
    BottomLeft,
    Left,
    TopLeft,
}

impl Direction {
    pub fn rotate_clockwise(self) -> Direction {
        match self {
            Direction::TopRight => Direction::Right,
            Direction::Right => Direction::BottomRight,
            Direction::BottomRight => Direction::BottomLeft,
            Direction::BottomLeft => Direction::Left,
            Direction::Left => Direction::TopLeft,
            Direction::TopLeft => Direction::TopRight,
        }
    }

    pub fn rotate_counter_clockwise(self) -> Direction {
        match self {
            Direction::TopRight => Direction::TopLeft,
            Direction::Right => Direction::TopRight,
            Direction::BottomRight => Direction::Right,
            Direction::BottomLeft => Direction::BottomRight,
            Direction::Left => Direction::BottomLeft,
            Direction::TopLeft => Direction::Left,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DirectionMap<T> {
    top_right: Option<T>,
    right: Option<T>,
    bottom_right: Option<T>,
    bottom_left: Option<T>,
    left: Option<T>,
    top_left: Option<T>,
}

impl<T> DirectionMap<T> {
    pub fn new() -> Self {
        DirectionMap {
            top_right: None,
            right: None,
            bottom_right: None,
            bottom_left: None,
            left: None,
            top_left: None,
        }
    }

    pub fn get(&self, direction: Direction) -> Option<&T> {
        match direction {
            Direction::TopRight => self.top_right.as_ref(),
            Direction::Right => self.right.as_ref(),
            Direction::BottomRight => self.bottom_right.as_ref(),
            Direction::BottomLeft => self.bottom_left.as_ref(),
            Direction::Left => self.left.as_ref(),
            Direction::TopLeft => self.top_left.as_ref(),
        }
    }

    pub fn set(&mut self, direction: Direction, value: T) {
        match direction {
            Direction::TopRight => self.top_right = Some(value),
            Direction::Right => self.right = Some(value),
            Direction::BottomRight => self.bottom_right = Some(value),
            Direction::BottomLeft => self.bottom_left = Some(value),
            Direction::Left => self.left = Some(value),
            Direction::TopLeft => self.top_left = Some(value),
        }
    }

    pub fn contains_key(&self, direction: Direction) -> bool {
        match direction {
            Direction::TopRight => self.top_right.is_some(),
            Direction::Right => self.right.is_some(),
            Direction::BottomRight => self.bottom_right.is_some(),
            Direction::BottomLeft => self.bottom_left.is_some(),
            Direction::Left => self.left.is_some(),
            Direction::TopLeft => self.top_left.is_some(),
        }
    }
}

impl<T> Default for DirectionMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct NeighborsIter {
    center: Position,
    state: Option<Direction>,
    dim: u8,
}

impl Iterator for NeighborsIter {
    type Item = (Direction, Position);

    fn next(&mut self) -> Option<Self::Item> {
        let Position(x, y) = self.center;
        loop {
            match self.state {
                None => return None,
                Some(Direction::TopLeft) => {
                    self.state = Some(Direction::TopRight);
                    if x == 0 {
                        self.state = Some(Direction::Right);
                        continue;
                    }

                    return Some((Direction::TopLeft, Position(x - 1, y)));
                }
                Some(Direction::TopRight) => {
                    self.state = Some(Direction::Right);
                    if x == 0 {
                        continue;
                    }

                    if y == self.dim {
                        self.state = Some(Direction::BottomRight);
                        continue;
                    }

                    return Some((Direction::TopRight, Position(x - 1, y + 1)));
                }
                Some(Direction::Right) => {
                    self.state = Some(Direction::BottomRight);
                    if y == self.dim {
                        continue;
                    }

                    return Some((Direction::Right, Position(x, y + 1)));
                }
                Some(Direction::BottomRight) => {
                    self.state = Some(Direction::BottomLeft);
                    if x == self.dim {
                        self.state = Some(Direction::Left);
                        continue;
                    }

                    return Some((Direction::BottomRight, Position(x + 1, y)));
                }
                Some(Direction::BottomLeft) => {
                    self.state = Some(Direction::Left);
                    if x == self.dim {
                        continue;
                    }

                    if y == 0 {
                        self.state = None;
                        return None;
                    }

                    return Some((Direction::BottomLeft, Position(x + 1, y - 1)));
                }
                Some(Direction::Left) => {
                    self.state = None;
                    if y == 0 {
                        continue;
                    }

                    return Some((Direction::Left, Position(x, y - 1)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_is_adjacent_0_0() {
        let a = Position(0, 0);
        assert!(a.is_adjacent(&Position(0, 1)));
        assert!(a.is_adjacent(&Position(1, 0)));
        // These should all be false.
        assert!(!a.is_adjacent(&Position(1, 1)));
        assert!(!a.is_adjacent(&Position(0, 2)));
        assert!(!a.is_adjacent(&Position(2, 0)));
        assert!(!a.is_adjacent(&Position(2, 2)));
    }

    #[test]
    fn position_is_adjacent_1_1() {
        let a = Position(1, 1);
        assert!(a.is_adjacent(&Position(0, 1)));
        assert!(a.is_adjacent(&Position(0, 2)));
        assert!(a.is_adjacent(&Position(1, 2)));
        assert!(a.is_adjacent(&Position(2, 0)));
        assert!(a.is_adjacent(&Position(2, 1)));
        assert!(a.is_adjacent(&Position(1, 0)));
        // These should all be false.
        assert!(!a.is_adjacent(&Position(0, 0)));
        assert!(!a.is_adjacent(&Position(2, 2)));
    }

    #[test]
    fn position_is_adjacent_2_2() {
        let a = Position(2, 2);
        assert!(a.is_adjacent(&Position(1, 2)));
        assert!(a.is_adjacent(&Position(1, 3)));
        assert!(a.is_adjacent(&Position(2, 3)));
        assert!(a.is_adjacent(&Position(2, 1)));
        assert!(a.is_adjacent(&Position(3, 1)));
        assert!(a.is_adjacent(&Position(3, 2)));
        // These should all be false.
        assert!(!a.is_adjacent(&Position(1, 1)));
        assert!(!a.is_adjacent(&Position(3, 3)));
    }

    #[test]
    fn neighbors_0_0() {
        let mut n = Position(0, 0).neighbors(u8::MAX);
        assert_eq!(n.next(), Some((Direction::Right, Position(0, 1))));
        assert_eq!(n.next(), Some((Direction::BottomRight, Position(1, 0))));
        assert_eq!(n.next(), None);
    }

    #[test]
    fn neighbors_1_1() {
        let mut n = Position(1, 1).neighbors(u8::MAX);
        assert_eq!(n.next(), Some((Direction::TopLeft, Position(0, 1))));
        assert_eq!(n.next(), Some((Direction::TopRight, Position(0, 2))));
        assert_eq!(n.next(), Some((Direction::Right, Position(1, 2))));
        assert_eq!(n.next(), Some((Direction::BottomRight, Position(2, 1))));
        assert_eq!(n.next(), Some((Direction::BottomLeft, Position(2, 0))));
        assert_eq!(n.next(), Some((Direction::Left, Position(1, 0))));
        assert_eq!(n.next(), None);
    }

    #[test]
    fn neighbors_2_3() {
        let mut n = Position(2, 3).neighbors(u8::MAX);
        assert_eq!(n.next(), Some((Direction::TopLeft, Position(1, 3))));
        assert_eq!(n.next(), Some((Direction::TopRight, Position(1, 4))));
        assert_eq!(n.next(), Some((Direction::Right, Position(2, 4))));
        assert_eq!(n.next(), Some((Direction::BottomRight, Position(3, 3))));
        assert_eq!(n.next(), Some((Direction::BottomLeft, Position(3, 2))));
        assert_eq!(n.next(), Some((Direction::Left, Position(2, 2))));
        assert_eq!(n.next(), None);
    }
}
