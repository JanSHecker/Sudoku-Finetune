mod generator;
mod grid;
mod solver;

pub use generator::{generate_one, GeneratedPuzzle};
pub use grid::Grid;
pub use solver::{rate, solve_count, SolveStats};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_puzzles_are_unique_and_match_the_solution() {
        let puzzle = generate_one(42, 100).unwrap();
        assert_eq!(solve_count(&puzzle.puzzle, 2), 1);
        assert!(puzzle.puzzle.is_valid());
        assert!(puzzle.solution.is_valid());
        for index in 0..81 {
            if puzzle.puzzle.cells()[index] != 0 {
                assert_eq!(puzzle.puzzle.cells()[index], puzzle.solution.cells()[index]);
            }
        }
    }

    #[test]
    fn generated_puzzles_are_reproducible() {
        let first = generate_one(987654321, 100).unwrap();
        let second = generate_one(987654321, 100).unwrap();
        assert_eq!(first.puzzle.to_line(), second.puzzle.to_line());
        assert_eq!(first.solution.to_line(), second.solution.to_line());
        assert_eq!(first.score, second.score);
    }
}
