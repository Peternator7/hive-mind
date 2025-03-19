use std::fmt;

/// Represents the color of a piece in the game.
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    /// Returns the opposing color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::Color;
    ///
    /// assert_eq!(Color::Black.opposing(), Color::White);
    /// assert_eq!(Color::White.opposing(), Color::Black);
    /// ```
    pub fn opposing(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

impl fmt::Display for Color {
    /// Formats the color as a string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::Color;
    ///
    /// assert_eq!(format!("{}", Color::Black), "Black");
    /// assert_eq!(format!("{}", Color::White), "White");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let color_str = match self {
            Color::Black => "Black",
            Color::White => "White",
        };
        write!(f, "{}", color_str)
    }
}

/// A map that associates values of type `T` with each color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorMap<T> {
    white: T,
    black: T,
}

impl<T: Default> ColorMap<T> {
    /// Creates a new `ColorMap` with the specified values for each color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::{Color, ColorMap};
    ///
    /// let map = ColorMap::new(1, 2);
    /// assert_eq!(*map.get(Color::White), 1);
    /// assert_eq!(*map.get(Color::Black), 2);
    /// ```
    pub fn new(white: T, black: T) -> Self {
        Self { white, black }
    }

    /// Returns a reference to the value associated with the specified color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::{Color, ColorMap};
    ///
    /// let map = ColorMap::new(1, 2);
    /// assert_eq!(*map.get(Color::White), 1);
    /// assert_eq!(*map.get(Color::Black), 2);
    /// ```
    pub fn get(&self, color: Color) -> &T {
        match color {
            Color::Black => &self.black,
            Color::White => &self.white,
        }
    }

    /// Returns a mutable reference to the value associated with the specified color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::{Color, ColorMap};
    ///
    /// let mut map = ColorMap::new(1, 2);
    /// *map.get_mut(Color::White) = 3;
    /// assert_eq!(*map.get(Color::White), 3);
    /// ```
    pub fn get_mut(&mut self, color: Color) -> &mut T {
        match color {
            Color::Black => &mut self.black,
            Color::White => &mut self.white,
        }
    }

    /// Sets the value associated with the specified color and returns the old value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::{ColorMap, Color};
    ///
    /// let mut map = ColorMap::new(1, 2);
    /// let old_value = map.set(Color::White, 3);
    /// assert_eq!(old_value, 1);
    /// assert_eq!(*map.get(Color::White), 3);
    /// ```
    pub fn set(&mut self, color: Color, value: T) -> T {
        let dst = self.get_mut(color);
        std::mem::replace(dst, value)
    }

    /// Returns a reference to the value associated with the black color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::ColorMap;
    ///
    /// let map = ColorMap::new(1, 2);
    /// assert_eq!(*map.black(), 2);
    /// ```
    pub fn black(&self) -> &T {
        &self.black
    }

    /// Returns a reference to the value associated with the white color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hive_engine::piece::ColorMap;
    ///
    /// let map = ColorMap::new(1, 2);
    /// assert_eq!(*map.white(), 1);
    /// ```
    pub fn white(&self) -> &T {
        &self.white
    }
}
