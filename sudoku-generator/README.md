# Sudoku Generator

This Rust crate generates difficult, uniquely solvable standard 9x9 Sudoku
puzzles for the fine-tuning experiments in the parent project.

## Build

```powershell
cargo build --release
```

## Usage

Generate reproducible puzzles as headerless TSV. Each line contains an
81-character puzzle followed by its 81-character solution; `0` means an empty
cell.

```powershell
cargo run --release -- --count 10 --seed 42 --min-score 1000 --output puzzles.tsv
```

Add `--metadata scores.jsonl` to write one JSON metadata record per puzzle for
the evaluation-set builders in `evaluation/`.

Use `--max-score` to select a bounded difficulty band, `--threads` to control
parallel generation, and `--max-attempts` to bound work. Difficulty is a
deterministic solver-search score, not a claim about human solving technique.

## Verification

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
