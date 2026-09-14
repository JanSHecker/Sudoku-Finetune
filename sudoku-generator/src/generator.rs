use crate::{rate, solve_count, Grid, SolveStats};

#[derive(Clone, Debug)]
pub struct GeneratedPuzzle {
    pub puzzle: Grid,
    pub solution: Grid,
    pub stats: SolveStats,
    pub score: u64,
}

#[derive(Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = (self.next_u64() as usize) % (index + 1);
            values.swap(index, other);
        }
    }
}

struct BoardBuilder {
    grid: Grid,
    rows: [u16; 9],
    columns: [u16; 9],
    boxes: [u16; 9],
    rng: Rng,
}

impl BoardBuilder {
    fn new(seed: u64) -> Self {
        Self {
            grid: Grid::empty(),
            rows: [0; 9],
            columns: [0; 9],
            boxes: [0; 9],
            rng: Rng::new(seed),
        }
    }

    fn candidates(&self, index: usize) -> u16 {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        0x03fe & !(self.rows[row] | self.columns[column] | self.boxes[box_index])
    }

    fn place(&mut self, index: usize, digit: u8) {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        let bit = 1u16 << digit;
        self.grid.cells[index] = digit;
        self.rows[row] |= bit;
        self.columns[column] |= bit;
        self.boxes[box_index] |= bit;
    }

    fn remove(&mut self, index: usize, digit: u8) {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        let bit = 1u16 << digit;
        self.grid.cells[index] = 0;
        self.rows[row] &= !bit;
        self.columns[column] &= !bit;
        self.boxes[box_index] &= !bit;
    }

    fn fill(&mut self, filled: usize) -> bool {
        if filled == 81 {
            return true;
        }

        let mut selected = None;
        let mut fewest = 10;
        for index in 0..81 {
            if self.grid.cells[index] != 0 {
                continue;
            }
            let candidates = self.candidates(index);
            let count = candidates.count_ones();
            if count < fewest {
                selected = Some((index, candidates));
                fewest = count;
            }
        }
        let Some((index, candidates)) = selected else {
            return false;
        };
        if candidates == 0 {
            return false;
        }

        let mut digits = [0u8; 9];
        let mut length = 0;
        for digit in 1..=9 {
            if candidates & (1u16 << digit) != 0 {
                digits[length] = digit;
                length += 1;
            }
        }
        self.rng.shuffle(&mut digits[..length]);
        for &digit in &digits[..length] {
            self.place(index, digit);
            if self.fill(filled + 1) {
                return true;
            }
            self.remove(index, digit);
        }
        false
    }

    fn build(mut self) -> Grid {
        assert!(self.fill(0));
        self.grid
    }
}

pub fn generate_one(seed: u64, min_score: u64) -> Option<GeneratedPuzzle> {
    let solution = BoardBuilder::new(seed).build();
    let mut puzzle = solution;
    let mut rng = Rng::new(seed ^ 0xa076_1d64_78bd_642f);
    let mut positions: Vec<usize> = (0..81).collect();
    rng.shuffle(&mut positions);

    for index in positions {
        let digit = puzzle.cells[index];
        puzzle.cells[index] = 0;
        if solve_count(&puzzle, 2) != 1 {
            puzzle.cells[index] = digit;
        }
    }

    let stats = rate(&puzzle)?;
    if stats.score < min_score {
        return None;
    }
    Some(GeneratedPuzzle {
        puzzle,
        solution,
        score: stats.score,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_builds_same_complete_board() {
        assert_eq!(
            BoardBuilder::new(1).build().to_line(),
            BoardBuilder::new(1).build().to_line()
        );
    }
}
