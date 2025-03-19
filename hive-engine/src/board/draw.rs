use crate::{piece::Insect, position::Position};

use super::Board;

use plotters::prelude::*;

pub fn draw_board(board: &Board, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dim = board.dim as usize;
    let diameter = 40;
    let radius = (diameter * 4) / 12;

    let vertical_offset = (3.0f64.sqrt() * radius as f64) as usize;

    // Create a 800*600 bitmap and start drawing
    let mut backend = SVGBackend::new(path, (1920, 1080));

    backend.draw_rect((0, 0), (1920, 1080), &RGBColor(196, 164, 152), true)?;

    for v in 0..dim {
        for h in 0..dim {
            let p = Position(v as u8, h as u8);
            let center = compute_center(v, h, diameter, vertical_offset);
            let pos = board.get(p).expect("out of bounds");

            if let Some(..) = pos.piece.get() {
                for (_, neighbor) in p.neighbors(dim as u8) {
                    let neighbor_center = compute_center(
                        neighbor.0 as usize,
                        neighbor.1 as usize,
                        diameter,
                        vertical_offset,
                    );

                    backend.draw_line(center, neighbor_center, &BLUE)?;
                }
            } else if pos.adjacent_pieces() > 0 {
                backend.draw_circle(center, radius as u32, &BLACK, false)?;
                let text_style = TextStyle::from(("sans-serif", 8).into_font()).color(&BLACK);
                backend.draw_text(
                    &format!("{},{}", v, h),
                    &text_style,
                    (center.0 - (radius as i32 / 2), center.1),
                )?;
            }
        }
    }

    for v in 0..dim {
        for h in 0..dim {
            let p = Position(v as u8, h as u8);
            let center = compute_center(v, h, diameter, vertical_offset);
            let pos = board.get(p).expect("out of bounds");

            if let Some(piece) = pos.piece.get() {
                let style;
                let text_style;
                if piece.is_white_piece() {
                    style = &WHITE;
                    text_style = TextStyle::from(("sans-serif", 20).into_font()).color(&BLACK);
                } else {
                    style = &BLACK;
                    text_style = TextStyle::from(("sans-serif", 20).into_font()).color(&WHITE);
                };

                backend.draw_circle(center, radius as u32, style, true)?;

                let is_frozen = board.frozen_map.get(Board::convert_coords_to_idx(v as u8,h as u8, dim as u8));
                let ring_style = if is_frozen { &RED } else { &YELLOW };

                backend.draw_circle(center, radius as u32, ring_style, false)?;

                let letter = match piece.role {
                    Insect::QueenBee => "🐝",
                    Insect::Spider => "🕷️",
                    Insect::Beetle => "🪲",
                    Insect::Grasshopper => "🦗",
                    Insect::SoldierAnt => "🐜",
                };

                let text_center = (center.0 - 11, center.1 - 8);
                backend.draw_text(letter, &text_style, text_center)?;
            } else if pos.adjacent_pieces() > 0 {
                backend.draw_circle(center, radius as u32, &BLACK, false)?;
            }
        }
    }

    // And if we want SVG backend
    // let backend = SVGBackend::new("output.svg", (800, 600));
    backend.present()?;
    Ok(())
}

fn compute_center(v: usize, h: usize, diameter: usize, vertical_offset: usize) -> (i32, i32) {
    let x = ((h + 1) * diameter) + (v * diameter / 2);
    let y = (v + 1) * vertical_offset;

    (x as i32, y as i32)
}
