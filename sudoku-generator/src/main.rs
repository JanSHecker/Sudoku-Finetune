use clap::Parser;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use sudoku_generator::generate_one;

#[derive(Debug, Parser)]
#[command(
    name = "sudoku-generator",
    version,
    about = "Generate difficult unique Sudoku puzzles"
)]
struct Args {
    /// Number of puzzles to emit.
    #[arg(short, long, default_value_t = 1)]
    count: usize,

    /// Master seed. Equal seeds produce equal output, independent of thread count.
    #[arg(long)]
    seed: Option<u64>,

    /// Minimum deterministic solver difficulty score.
    #[arg(long, default_value_t = 1000)]
    min_score: u64,

    /// Optional inclusive upper bound for the deterministic solver score.
    #[arg(long)]
    max_score: Option<u64>,

    /// Maximum candidate attempts before failing.
    #[arg(long)]
    max_attempts: Option<usize>,

    /// Number of worker threads used for candidate generation.
    #[arg(long)]
    threads: Option<usize>,

    /// Number of candidates generated per parallel batch.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Output path for TSV output. The default writes to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional JSONL metadata path for emitted records.
    #[arg(long)]
    metadata: Option<PathBuf>,
}

fn derived_seed(master: u64, attempt: usize) -> u64 {
    let mut value = master ^ (attempt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn run(args: Args) -> Result<(), String> {
    if args.count == 0 {
        return Ok(());
    }
    if args.batch_size == 0 {
        return Err("--batch-size must be greater than zero".to_owned());
    }
    if args
        .max_score
        .is_some_and(|max_score| max_score < args.min_score)
    {
        return Err("--max-score must be greater than or equal to --min-score".to_owned());
    }
    if let Some(threads) = args.threads {
        if threads == 0 {
            return Err("--threads must be greater than zero".to_owned());
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("could not configure worker threads: {error}"))?;
    }

    let seed = args.seed.unwrap_or_else(default_seed);
    let max_attempts = args
        .max_attempts
        .unwrap_or_else(|| args.count.saturating_mul(10_000).max(args.count));
    let mut records = Vec::with_capacity(args.count);
    let mut seen = HashSet::with_capacity(args.count);
    let mut start = 0usize;

    while start < max_attempts && records.len() < args.count {
        let end = start.saturating_add(args.batch_size).min(max_attempts);
        let mut candidates: Vec<_> = (start..end)
            .into_par_iter()
            .filter_map(|attempt| {
                let attempt_seed = derived_seed(seed, attempt);
                generate_one(attempt_seed, args.min_score)
                    .filter(|puzzle| args.max_score.is_none_or(|max| puzzle.score <= max))
                    .map(|puzzle| (attempt, attempt_seed, puzzle))
            })
            .collect();
        candidates.sort_unstable_by_key(|(attempt, _, _)| *attempt);
        for (attempt, attempt_seed, puzzle) in candidates {
            if seen.insert(puzzle.puzzle.to_line()) {
                records.push((attempt, attempt_seed, puzzle));
            }
            if records.len() == args.count {
                break;
            }
        }
        records.truncate(args.count);
        start = end;
    }

    if records.len() != args.count {
        return Err(format!(
            "generated {} of {} requested puzzles after {} attempts; lower --min-score or raise --max-attempts",
            records.len(), args.count, max_attempts
        ));
    }

    let writer: Box<dyn Write> = match args.output {
        Some(path) => Box::new(
            File::create(&path)
                .map_err(|error| format!("could not create {}: {error}", path.display()))?,
        ),
        None => Box::new(io::stdout()),
    };
    let mut writer = BufWriter::new(writer);
    for (_, _, puzzle) in &records {
        writeln!(
            writer,
            "{}\t{}",
            puzzle.puzzle.to_line(),
            puzzle.solution.to_line()
        )
        .map_err(|error| format!("could not write output: {error}"))?;
        eprintln!("generated puzzle score={}", puzzle.score);
    }
    writer
        .flush()
        .map_err(|error| format!("could not flush output: {error}"))?;

    if let Some(path) = args.metadata {
        let file = File::create(&path)
            .map_err(|error| format!("could not create {}: {error}", path.display()))?;
        let mut metadata = BufWriter::new(file);
        for (index, (attempt, attempt_seed, puzzle)) in records.iter().enumerate() {
            let clues = puzzle
                .puzzle
                .cells()
                .iter()
                .filter(|&&cell| cell != 0)
                .count();
            let record = json!({
                "schema": "sudoku-generator-v1",
                "id": format!("candidate-{index:08}"),
                "attempt": attempt,
                "seed": attempt_seed,
                "puzzle": puzzle.puzzle.to_line(),
                "solution": puzzle.solution.to_line(),
                "score": puzzle.score,
                "clues": clues,
                "nodes": puzzle.stats.nodes,
                "backtracks": puzzle.stats.backtracks,
                "guesses": puzzle.stats.guesses,
                "max_depth": puzzle.stats.max_depth,
            });
            serde_json::to_writer(&mut metadata, &record)
                .map_err(|error| format!("could not serialize metadata: {error}"))?;
            metadata
                .write_all(b"\n")
                .map_err(|error| format!("could not write metadata: {error}"))?;
        }
        metadata
            .flush()
            .map_err(|error| format!("could not flush metadata: {error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
