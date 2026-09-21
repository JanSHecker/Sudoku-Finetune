"""Create a separate deduction dataset with representation-v2 state rendering.

The original deduction artifacts are preserved.  This converter changes only
the user prompt's state rendering: it adds row/column labels, box separators,
and the same bracketed candidate-grid format used by the representation
adapter. Targets, rows, splits, and answer metadata remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORMAT_NAME = "sudoku-representation-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("schema") != "sudoku-deduction-example"
                or row.get("schema_version") != 1
                or not isinstance(row.get("id"), str)
                or not isinstance(row.get("state"), dict)
                or not isinstance(row["state"].get("board"), str)
                or not isinstance(row["state"].get("candidate_grid"), list)
                or len(row["state"]["candidate_grid"]) != 9
            ):
                raise ValueError(f"invalid deduction row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def candidate_values(row: dict[str, Any]) -> tuple[str, list[str]]:
    board = row["state"]["board"]
    if len(board) != 81 or any(character not in "0123456789" for character in board):
        raise ValueError(f"invalid board for {row['id']}")
    values: list[str] = []
    for line in row["state"]["candidate_grid"]:
        if not isinstance(line, str) or len(line.split()) != 9:
            raise ValueError(f"invalid candidate grid for {row['id']}")
        for token in line.split():
            value = token[1:-1] if token.startswith("[") and token.endswith("]") else token
            if not value or any(character not in "123456789" for character in value):
                raise ValueError(f"invalid candidate value for {row['id']}")
            values.append(value)
    if len(values) != 81:
        raise ValueError(f"candidate grid does not contain 81 cells for {row['id']}")
    return board, values


def render_grid(board: str, values: list[str]) -> str:
    rendered = [
        value if board[cell] != "0" else f"[{value}]"
        for cell, value in enumerate(values)
    ]
    lines = ["      c1 c2 c3 | c4 c5 c6 | c7 c8 c9"]
    for row in range(9):
        tokens = rendered[row * 9 : row * 9 + 9]
        lines.append(
            f"r{row + 1}:  "
            + " ".join(tokens[:3])
            + " | "
            + " ".join(tokens[3:6])
            + " | "
            + " ".join(tokens[6:])
        )
        if row in (2, 5):
            lines.append("     ---------+---------+---------")
    return "\n".join(lines)


def deduction_prompt(board: str, values: list[str]) -> str:
    return (
        "Identify one sound Sudoku deduction from the state below. "
        "Return JSON containing the rule, witness, placements, and/or "
        "candidate eliminations. Do not guess.\n\n"
        "State:\n"
        + render_grid(board, values)
    )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    for row in rows:
        board, values = candidate_values(row)
        converted = dict(row)
        converted["prompt"] = deduction_prompt(board, values)
        converted["prompt_format"] = FORMAT_NAME
        output.append(converted)
        prompt_lengths.append(len(converted["prompt"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema": "sudoku-deduction-prompt-format-report",
        "schema_version": 1,
        "input": str(args.input),
        "output": str(args.output),
        "format": FORMAT_NAME,
        "rows": len(output),
        "rows_by_split": dict(sorted(Counter(row.get("split") for row in output).items())),
        "puzzles_by_split": {
            split: len({row["puzzle"] for row in output if row.get("split") == split})
            for split in sorted({row.get("split") for row in output})
        },
        "prompt_characters": {
            "minimum": min(prompt_lengths),
            "maximum": max(prompt_lengths),
            "average": sum(prompt_lengths) / len(prompt_lengths),
        },
        "target_preserved": True,
        "old_dataset_preserved": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
