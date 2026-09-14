"""Build balanced, state-level Sudoku deduction examples from Rust traces.

The source traces contain the hidden solution for validation, but the emitted
training records deliberately contain only the current state, candidate grid,
and one target deduction.  Alternative deductions are retained as labels so a
validator or RLVR evaluator can accept any sound conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALL_DIGITS = set("123456789")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--per-rule", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Mark every selected example as benchmark instead of hashing into splits.",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        action="append",
        help="Evaluation metadata JSONL files whose puzzles must be excluded.",
    )
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if record.get("schema") != "sudoku-generator-trace":
                    raise ValueError(f"unsupported trace schema at {path}:{line_number}")
                records.append(record)
    if not records:
        raise ValueError("no trace records were loaded")
    return records


def read_heldout(paths: list[Path]) -> set[str]:
    puzzles: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                puzzle = row.get("puzzle")
                if not isinstance(puzzle, str) or len(puzzle) != 81:
                    raise ValueError(f"invalid held-out puzzle at {path}:{line_number}")
                puzzles.add(puzzle)
    return puzzles


def rule_key(deduction: dict[str, Any]) -> str:
    technique = deduction.get("technique", {})
    kind = technique.get("kind")
    if kind != "fish":
        return str(kind)
    return f"fish_{technique['size']}_{technique['orientation']}"


def candidate_grid_lines(board: str, values: list[str]) -> list[str]:
    if len(values) != 81:
        raise ValueError("candidate grid must contain 81 cells")
    rendered = [
        value if board[index] != "0" else f"[{value}]"
        for index, value in enumerate(values)
    ]
    return [" ".join(rendered[index : index + 9]) for index in range(0, 81, 9)]


def parse_grid(value: str, name: str) -> list[str]:
    if len(value) != 81 or any(character not in "0123456789" for character in value):
        raise ValueError(f"{name} must contain 81 ASCII digits")
    return list(value)


def parse_candidate_grid(board: list[str], values: list[str]) -> list[set[str]]:
    if len(values) != 81:
        raise ValueError("candidate grid must contain 81 cells")
    candidates: list[set[str]] = []
    for index, value in enumerate(values):
        if board[index] != "0":
            if value != board[index]:
                raise ValueError(f"filled cell {index} disagrees with candidate grid")
            candidates.append(set())
            continue
        if not value or any(character not in ALL_DIGITS for character in value):
            raise ValueError(f"invalid candidates at cell {index}: {value!r}")
        if len(set(value)) != len(value) or value != "".join(sorted(value)):
            raise ValueError(f"candidates at cell {index} are not sorted and unique")
        candidates.append(set(value))
    return candidates


def peers(cell: int) -> set[int]:
    row, column = divmod(cell, 9)
    box_row, box_column = row // 3, column // 3
    result = {row * 9 + offset for offset in range(9)}
    result.update(offset * 9 + column for offset in range(9))
    result.update(
        (box_row * 3 + local_row) * 9 + box_column * 3 + local_column
        for local_row in range(3)
        for local_column in range(3)
    )
    result.discard(cell)
    return result


def apply_deduction(
    board_value: str,
    candidate_values: list[str],
    deduction: dict[str, Any],
) -> tuple[str, list[str]]:
    board = parse_grid(board_value, "board")
    candidates = parse_candidate_grid(board, candidate_values)

    for placement in deduction.get("placements", []):
        cell = placement.get("cell")
        digit = str(placement.get("digit"))
        if not isinstance(cell, int) or not 0 <= cell < 81 or digit not in ALL_DIGITS:
            raise ValueError("invalid placement")
        if board[cell] != "0" or digit not in candidates[cell]:
            raise ValueError(f"placement {cell}={digit} is not a current candidate")
        if any(board[peer] == digit for peer in peers(cell)):
            raise ValueError(f"placement {cell}={digit} conflicts with a peer")
        board[cell] = digit
        candidates[cell] = set()
        for peer in peers(cell):
            if board[peer] == "0" and digit in candidates[peer]:
                candidates[peer].remove(digit)
                if not candidates[peer]:
                    raise ValueError(f"placement {cell}={digit} empties candidate cell {peer}")

    for elimination in deduction.get("eliminations", []):
        cell = elimination.get("cell")
        digit = str(elimination.get("digit"))
        if not isinstance(cell, int) or not 0 <= cell < 81 or digit not in ALL_DIGITS:
            raise ValueError("invalid elimination")
        if board[cell] != "0" or digit not in candidates[cell]:
            raise ValueError(f"elimination {cell}!={digit} is not a current candidate")
        candidates[cell].remove(digit)
        if not candidates[cell]:
            raise ValueError(f"elimination empties candidate cell {cell}")

    values = [
        board[index] if board[index] != "0" else "".join(sorted(candidates[index]))
        for index in range(81)
    ]
    return "".join(board), values


def candidate_fingerprint(values: list[str]) -> str:
    return hashlib.sha256("|".join(values).encode()).hexdigest()


def split_for_puzzle(puzzle: str) -> str:
    bucket = int(hashlib.sha256(puzzle.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def deduction_prompt(board: str, candidate_values: list[str]) -> str:
    return (
        "Identify one sound Sudoku deduction from the state below. "
        "Return JSON containing the rule, witness, placements, and/or "
        "candidate eliminations. Do not guess.\n\n"
        "State:\n"
        + "\n".join(
            f"r{row + 1}: {line}"
            for row, line in enumerate(candidate_grid_lines(board, candidate_values))
        )
    )


def expand_records(
    source_records: list[dict[str, Any]], heldout: set[str], benchmark: bool
) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    excluded = 0
    for source in source_records:
        puzzle = source["puzzle"]
        if puzzle in heldout:
            excluded += 1
            continue
        solution = source["solution"]
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        for step_index, step in enumerate(source.get("steps", [])):
            board = step.get("board_before")
            candidate_values = step.get("candidate_grid_before")
            available = step.get("available_deductions", [])
            if not isinstance(board, str) or not isinstance(candidate_values, list):
                raise ValueError(f"step {step_index} has no complete before-state")
            if not available:
                available = [
                    {
                        "technique": step["technique"],
                        "placements": step.get("placements", []),
                        "eliminations": step.get("eliminations", []),
                        "witness": step.get("witness", {"none": None}),
                    }
                ]
            valid: list[dict[str, Any]] = []
            for deduction in available:
                valid.append(deduction)
            for target_index, target in enumerate(valid):
                after_board, after_candidates = apply_deduction(board, candidate_values, target)
                key = f"{source['id']}:{step_index}:{target_index}"
                prompt = deduction_prompt(board, candidate_values)
                examples.append(
                    {
                        "schema": "sudoku-deduction-example",
                        "schema_version": 1,
                        "id": f"deduction-{hashlib.sha256(key.encode()).hexdigest()[:16]}",
                        "source_trace": source["id"],
                        "split": "benchmark" if benchmark else split_for_puzzle(puzzle),
                        "source_step": step_index,
                        "puzzle": puzzle,
                        "solution_sha256": solution_hash,
                        "score": source.get("score"),
                        "clues": source.get("clues"),
                        "state": {
                            "board": board,
                            "candidate_grid": candidate_grid_lines(board, candidate_values),
                        },
                        "target_rule": rule_key(target),
                        "target_deduction": target,
                        "prompt": prompt,
                        "completion": json.dumps(
                            target, sort_keys=True, separators=(",", ":")
                        ),
                        "valid_deduction_count": len(valid),
                        "valid_deductions": valid,
                        "next_state": {
                            "board": after_board,
                            "candidate_grid": candidate_grid_lines(after_board, after_candidates),
                        },
                    }
                )
    return examples, excluded


def select_balanced(examples: list[dict[str, Any]], per_rule: int, seed: int) -> list[dict[str, Any]]:
    if per_rule <= 0:
        raise ValueError("--per-rule must be greater than zero")
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_rule[example["target_rule"]].append(example)
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for rule in sorted(by_rule):
        rows = by_rule[rule][:]
        rng.shuffle(rows)
        selected.extend(rows[:per_rule])
    selected.sort(key=lambda row: row["id"])
    return selected


def main() -> None:
    args = parse_args()
    source = read_jsonl(args.input)
    heldout = read_heldout(args.heldout or [])
    expanded, excluded = expand_records(source, heldout, args.benchmark)
    selected = select_balanced(expanded, args.per_rule, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for example in selected:
            stream.write(json.dumps(example, sort_keys=True) + "\n")

    available = Counter(example["target_rule"] for example in expanded)
    chosen = Counter(example["target_rule"] for example in selected)
    report = {
        "schema": "sudoku-deduction-dataset-report",
        "schema_version": 1,
        "source_records": len(source),
        "heldout_puzzles": len(heldout),
        "source_records_excluded": excluded,
        "expanded_examples": len(expanded),
        "selected_examples": len(selected),
        "available_by_rule": dict(sorted(available.items())),
        "selected_by_rule": dict(sorted(chosen.items())),
        "source_puzzles": len({example["source_trace"] for example in selected}),
        "selected_by_split": dict(
            sorted(Counter(example["split"] for example in selected).items())
        ),
        "benchmark": args.benchmark,
        "prompt_characters": {
            "minimum": min((len(example["prompt"]) for example in selected), default=0),
            "maximum": max((len(example["prompt"]) for example in selected), default=0),
            "average": (
                sum(len(example["prompt"]) for example in selected) / len(selected)
                if selected
                else 0
            ),
        },
        "candidate_grid_format": "nine rows; fixed cells are digits and empty cells are [candidate digits]",
        "solution_exposure": "solution omitted; only solution_sha256 is retained",
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
