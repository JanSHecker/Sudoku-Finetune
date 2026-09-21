"""Audit overlap between Sudoku training artifacts and benchmark artifacts.

The audit is intentionally conservative: exact puzzle/state matches are the
primary leakage signal, while digit-renaming normalization catches Sudoku
isomorphs that differ only by a permutation of 1..9.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deduction-training", type=Path, required=True)
    parser.add_argument("--deduction-benchmark", type=Path, required=True)
    parser.add_argument("--deduction-benchmark-traces", type=Path)
    parser.add_argument("--final-benchmark", type=Path, required=True)
    parser.add_argument("--representation-training", type=Path)
    parser.add_argument("--representation-benchmark", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def digit_normalized(value: str) -> str:
    """Canonicalize a puzzle under a global permutation of its digit labels."""
    labels: dict[str, str] = {}
    next_digit = 1
    result: list[str] = []
    for character in value:
        if character == "0":
            result.append(character)
            continue
        if character not in labels:
            labels[character] = str(next_digit)
            next_digit += 1
        result.append(labels[character])
    return "".join(result)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def puzzle(row: dict[str, Any], representation: bool = False) -> str | None:
    value = row.get("source_puzzle") if representation else row.get("puzzle")
    return value if isinstance(value, str) else None


def state_key(row: dict[str, Any]) -> str | None:
    state = row.get("state")
    if not isinstance(state, dict):
        return None
    board = state.get("board")
    candidates = state.get("candidate_grid")
    if not isinstance(board, str) or not isinstance(candidates, list):
        return None
    return json.dumps([board, candidates], sort_keys=True, separators=(",", ":"))


def values(rows: Iterable[dict[str, Any]], key_function: Any) -> set[str]:
    return {value for row in rows if (value := key_function(row)) is not None}


def row_ids(rows: Iterable[dict[str, Any]], key_function: Any, matches: set[str]) -> list[str]:
    result: list[str] = []
    for row in rows:
        key = key_function(row)
        if key in matches:
            result.append(str(row.get("id", "<missing-id>")))
    return result


def overlap_summary(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    left_representation: bool = False,
    right_representation: bool = False,
) -> dict[str, Any]:
    left_puzzle = lambda row: puzzle(row, left_representation)
    right_puzzle = lambda row: puzzle(row, right_representation)
    exact = values(left, left_puzzle) & values(right, right_puzzle)
    normalized_left = {digit_normalized(value) for value in values(left, left_puzzle)}
    normalized_right = {digit_normalized(value) for value in values(right, right_puzzle)}
    normalized = normalized_left & normalized_right

    result: dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "left_rows": len(left),
        "right_rows": len(right),
        "left_puzzles": len(values(left, left_puzzle)),
        "right_puzzles": len(values(right, right_puzzle)),
        "exact_puzzle_overlap": len(exact),
        "digit_normalized_puzzle_overlap": len(normalized),
        "exact_puzzle_examples": sorted(exact)[:5],
        "digit_normalized_puzzle_examples": sorted(normalized)[:5],
    }

    left_states = values(left, state_key)
    right_states = values(right, state_key)
    exact_states = left_states & right_states
    result["exact_state_overlap"] = len(exact_states)
    result["exact_state_examples"] = sorted(exact_states)[:3]

    left_prompts = values(left, lambda row: compact_text(row["prompt"]) if isinstance(row.get("prompt"), str) else None)
    right_prompts = values(right, lambda row: compact_text(row["prompt"]) if isinstance(row.get("prompt"), str) else None)
    exact_prompts = left_prompts & right_prompts
    result["normalized_prompt_overlap"] = len(exact_prompts)
    result["normalized_prompt_examples"] = sorted(exact_prompts)[:3]

    left_hashes = values(left, lambda row: row.get("solution_sha256") if isinstance(row.get("solution_sha256"), str) else None)
    right_hashes = values(right, lambda row: row.get("solution_sha256") if isinstance(row.get("solution_sha256"), str) else None)
    result["solution_hash_overlap"] = len(left_hashes & right_hashes)

    if exact:
        result["left_ids_for_exact_puzzles"] = row_ids(left, left_puzzle, exact)[:10]
        result["right_ids_for_exact_puzzles"] = row_ids(right, right_puzzle, exact)[:10]
    return result


def main() -> None:
    args = parse_args()
    deduction_training = read_jsonl(args.deduction_training)
    deduction_benchmark = read_jsonl(args.deduction_benchmark)
    deduction_benchmark_traces = (
        read_jsonl(args.deduction_benchmark_traces)
        if args.deduction_benchmark_traces
        else []
    )
    final_benchmark = read_jsonl(args.final_benchmark)
    representation_training = (
        read_jsonl(args.representation_training) if args.representation_training else []
    )
    representation_benchmark = (
        [
            row
            for row in read_jsonl(args.representation_benchmark)
            if row.get("split") == "test"
        ]
        if args.representation_benchmark
        else []
    )

    training_rows = [row for row in deduction_training if row.get("split") == "train"]
    training_all = deduction_training
    representation_train = [row for row in representation_training if row.get("split") == "train"]

    comparisons = [
        overlap_summary(
            training_rows,
            deduction_benchmark,
            "deduction train",
            "deduction benchmark",
        ),
        overlap_summary(
            training_all,
            deduction_benchmark,
            "deduction all splits",
            "deduction benchmark",
        ),
        overlap_summary(
            training_rows,
            final_benchmark,
            "deduction train",
            "final benchmark",
        ),
        overlap_summary(
            training_all,
            final_benchmark,
            "deduction all splits",
            "final benchmark",
        ),
    ]
    if representation_training:
        comparisons.extend(
            [
                overlap_summary(
                    representation_train,
                    deduction_benchmark,
                    "representation train",
                    "deduction benchmark",
                    left_representation=True,
                ),
                overlap_summary(
                    representation_train,
                    final_benchmark,
                    "representation train",
                    "final benchmark",
                    left_representation=True,
                ),
            ]
        )
    if representation_benchmark:
        comparisons.append(
            overlap_summary(
                representation_train,
                representation_benchmark,
                "representation train",
                "representation benchmark",
                left_representation=True,
                right_representation=True,
            )
        )

    if deduction_benchmark_traces:
        comparisons.append(
            overlap_summary(
                training_all,
                deduction_benchmark_traces,
                "deduction all splits",
                "deduction benchmark traces",
            )
        )

    deduction_by_split = {
        split: [row for row in deduction_training if row.get("split") == split]
        for split in ("train", "validation", "test")
    }
    representation_by_split = {
        split: [row for row in representation_training if row.get("split") == split]
        for split in ("train", "validation", "test")
    }

    def split_puzzle_overlap(groups: dict[str, list[dict[str, Any]]], representation: bool) -> dict[str, int]:
        result: dict[str, int] = {}
        for index, left_name in enumerate(("train", "validation", "test")):
            for right_name in ("train", "validation", "test")[index + 1 :]:
                left = values(groups[left_name], lambda row: puzzle(row, representation))
                right = values(groups[right_name], lambda row: puzzle(row, representation))
                result[f"{left_name}_vs_{right_name}"] = len(left & right)
        return result

    source_counts: dict[str, int] = defaultdict(int)
    for row in deduction_training:
        source_counts[str(row.get("source_trace", "<missing>"))] += 1
    result = {
        "schema": "sudoku-dataset-overlap-report",
        "training_rows": len(training_rows),
        "training_all_rows": len(training_all),
        "training_puzzles": len(values(training_rows, lambda row: puzzle(row))),
        "deduction_benchmark_rows": len(deduction_benchmark),
        "final_benchmark_rows": len(final_benchmark),
        "representation_training_rows": len(representation_training),
        "representation_train_rows": len(representation_train),
        "deduction_internal_split_puzzle_overlap": split_puzzle_overlap(
            deduction_by_split, False
        ),
        "representation_internal_split_puzzle_overlap": split_puzzle_overlap(
            representation_by_split, True
        ),
        "comparisons": comparisons,
        "deduction_training_source_trace_count": len(source_counts),
        "deduction_training_source_rows_top": sorted(
            source_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
