use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HiveError {
    PositionOutOfBounds,
    PositionIsEmpty,
    PositionIsFrozen,
    DeserializationError(DeserializationError),
    ValidationError(ValidationError),
    ZeroPositionIsUnplayable,
    InvalidPlacementLocation(InvalidPlacementLocation),
}

impl Display for HiveError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            HiveError::PositionOutOfBounds => write!(f, "PositionOutOfBounds"),
            HiveError::PositionIsEmpty => write!(f, "PositionIsEmpty"),
            HiveError::PositionIsFrozen => write!(f, "PositionIsFrozen"),
            HiveError::DeserializationError(e) => write!(f, "DeserializationError({})", e),
            HiveError::ValidationError(e) => write!(f, "ValidationError({})", e),
            HiveError::ZeroPositionIsUnplayable => write!(f, "ZeroPositionIsUnplayable"),
            HiveError::InvalidPlacementLocation(e) => write!(f, "InvalidPlacementLocation({})", e),
        }
    }
}

impl From<InvalidPlacementLocation> for HiveError {
    fn from(error: InvalidPlacementLocation) -> Self {
        HiveError::InvalidPlacementLocation(error)
    }
}

impl From<ValidationError> for HiveError {
    fn from(error: ValidationError) -> Self {
        HiveError::ValidationError(error)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InvalidPlacementLocation {
    AdjacentToOppositeColor,
    NotAdjacentToOwnColor,
    PositionOccupied,
    SecondPieceMustBeAdjacentToFirst,
}

impl Display for InvalidPlacementLocation {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            InvalidPlacementLocation::AdjacentToOppositeColor => {
                write!(f, "AdjacentToOppositeColor")
            }
            InvalidPlacementLocation::NotAdjacentToOwnColor => write!(f, "NotAdjacentToOwnColor"),
            InvalidPlacementLocation::PositionOccupied => write!(f, "PositionOccupied"),
            InvalidPlacementLocation::SecondPieceMustBeAdjacentToFirst => {
                write!(f, "SecondPieceMustBeAdjacentToFirst")
            }
        }
    }
}

impl Error for HiveError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidationError {
    MultipleHives,
    WhitePiecesAreMissingFromLL,
    BlackPiecesAreMissingFromLL,
    AvailableTilesAreMissingFromLL,
    TileAdjacencyCountsAreWrong,
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            ValidationError::MultipleHives => write!(f, "MultipleHives"),
            ValidationError::WhitePiecesAreMissingFromLL => {
                write!(f, "WhitePiecesAreMissingFromLL")
            }
            ValidationError::BlackPiecesAreMissingFromLL => {
                write!(f, "BlackPiecesAreMissingFromLL")
            }
            ValidationError::AvailableTilesAreMissingFromLL => {
                write!(f, "AvailableTilesAreMissingFromLL")
            }
            ValidationError::TileAdjacencyCountsAreWrong => {
                write!(f, "TileAdjacencyCountsAreWrong")
            }
        }
    }
}

/// Represents an error that occurs during deserialization.
///
/// This error is used when an invalid character is encountered during the deserialization process.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeserializationError {
    EndOfStream,
    InvalidChar(char, char),
}

impl Display for DeserializationError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            DeserializationError::InvalidChar(role, id) => {
                write!(f, "InvalidChar({},{})", role, id)
            }
            DeserializationError::EndOfStream => write!(f, "EndOfStream"),
        }
    }
}
