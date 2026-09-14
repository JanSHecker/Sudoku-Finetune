use crate::Grid;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SolveStats {
    pub solutions: u8,
    pub nodes: u64,
    pub backtracks: u64,
    pub guesses: u64,
    pub max_depth: u8,
    pub score: u64,
}

struct Solver {
    grid: [u8; 81],
    rows: [u16; 9],
    columns: [u16; 9],
    boxes: [u16; 9],
    stats: SolveStats,
    limit: u8,
}

impl Solver {
    fn new(grid: &Grid, limit: u8) -> Option<Self> {
        let mut solver = Self {
            grid: *grid.cells(),
            rows: [0; 9],
            columns: [0; 9],
            boxes: [0; 9],
            stats: SolveStats::default(),
            limit,
        };

        for index in 0..81 {
            let digit = solver.grid[index];
            if digit == 0 {
                continue;
            }
            if !(1..=9).contains(&digit) || !solver.place(index, digit) {
                return None;
            }
        }
        Some(solver)
    }

    fn place(&mut self, index: usize, digit: u8) -> bool {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        let bit = 1u16 << digit;
        if self.rows[row] & bit != 0
            || self.columns[column] & bit != 0
            || self.boxes[box_index] & bit != 0
        {
            return false;
        }
        self.grid[index] = digit;
        self.rows[row] |= bit;
        self.columns[column] |= bit;
        self.boxes[box_index] |= bit;
        true
    }

    fn remove(&mut self, index: usize, digit: u8) {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        let bit = 1u16 << digit;
        self.grid[index] = 0;
        self.rows[row] &= !bit;
        self.columns[column] &= !bit;
        self.boxes[box_index] &= !bit;
    }

    fn candidates(&self, index: usize) -> u16 {
        let row = index / 9;
        let column = index % 9;
        let box_index = (row / 3) * 3 + column / 3;
        0x03fe & !(self.rows[row] | self.columns[column] | self.boxes[box_index])
    }

    fn next_cell(&self) -> Option<(usize, u16)> {
        let mut best = None;
        let mut best_count = 10;
        for index in 0..81 {
            if self.grid[index] != 0 {
                continue;
            }
            let candidates = self.candidates(index);
            let count = candidates.count_ones();
            if count < best_count {
                best = Some((index, candidates));
                best_count = count;
                if count <= 1 {
                    break;
                }
            }
        }
        best
    }

    fn search(&mut self, depth: u8) {
        if self.stats.solutions >= self.limit {
            return;
        }
        self.stats.nodes += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);

        let Some((index, candidates)) = self.next_cell() else {
            self.stats.solutions += 1;
            return;
        };
        if candidates == 0 {
            self.stats.backtracks += 1;
            return;
        }
        if candidates.count_ones() > 1 {
            self.stats.guesses += 1;
        }

        let mut remaining = candidates;
        while remaining != 0 && self.stats.solutions < self.limit {
            let bit = remaining & remaining.wrapping_neg();
            let digit = bit.trailing_zeros() as u8;
            remaining &= !bit;
            self.place(index, digit);
            self.search(depth + 1);
            self.remove(index, digit);
        }
    }
}

pub fn solve_count(grid: &Grid, limit: u8) -> u8 {
    if limit == 0 {
        return 0;
    }
    let Some(mut solver) = Solver::new(grid, limit) else {
        return 0;
    };
    solver.search(0);
    solver.stats.solutions
}

pub fn rate(grid: &Grid) -> Option<SolveStats> {
    let mut solver = Solver::new(grid, 2)?;
    solver.search(0);
    if solver.stats.solutions != 1 {
        return None;
    }
    solver.stats.score = solver.stats.nodes
        + solver.stats.backtracks * 10
        + solver.stats.guesses * 5
        + u64::from(solver.stats.max_depth) * 20;
    Some(solver.stats)
}
