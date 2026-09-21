"""Build auxiliary Sudoku representation-learning examples.

The examples use only the visible board and candidate grid from the existing
state-level deduction dataset.  They deliberately omit deductions and hidden
solutions.  Their purpose is to teach coordinate, unit, peer, and candidate
lookup before deduction fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ALL_DIGITS = set("123456789")
SCHEMA_VERSION = 2
TASKS = (
    "cell_lookup",
    "coordinate_lookup",
    "unit_cells",
    "candidate_locations",
    "peer_relation",
    "naked_single_scan",
)
V2_TASK_PLAN = (
    "cell_lookup",
    "coordinate_lookup",
    "unit_cells",
    "candidate_locations",
    "peer_relation",
    "naked_single_scan",
    "coordinate_lookup",
    "candidate_locations",
)
PEER_RELATIONS = ("same_row", "same_column", "same_box", "disjoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tasks-per-state", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format-version",
        type=int,
        choices=(1, 2),
        default=2,
        help="Representation dataset format; v1 preserves the original sampler.",
    )
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("schema") != "sudoku-deduction-example"
                    or row.get("schema_version") != 1
                    or not isinstance(row.get("puzzle"), str)
                    or len(row["puzzle"]) != 81
                    or any(character not in "0123456789" for character in row["puzzle"])
                    or row.get("split") not in {"train", "validation", "test"}
                    or not isinstance(row.get("state"), dict)
                ):
                    raise ValueError(f"invalid deduction row at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise ValueError("no deduction rows were loaded")
    return rows


def parse_state(row: dict[str, Any]) -> tuple[str, list[str]]:
    state = row["state"]
    board = state.get("board")
    lines = state.get("candidate_grid")
    if (
        not isinstance(board, str)
        or len(board) != 81
        or any(character not in "0123456789" for character in board)
        or not isinstance(lines, list)
        or len(lines) != 9
    ):
        raise ValueError(f"invalid state for {row.get('id')}")

    values: list[str] = []
    for line in lines:
        if not isinstance(line, str) or len(line.split()) != 9:
            raise ValueError(f"invalid candidate grid for {row.get('id')}")
        for token in line.split():
            value = token[1:-1] if token.startswith("[") and token.endswith("]") else token
            if not value or any(character not in ALL_DIGITS for character in value):
                raise ValueError(f"invalid candidate value for {row.get('id')}")
            if len(set(value)) != len(value) or value != "".join(sorted(value)):
                raise ValueError(f"candidate value is not sorted for {row.get('id')}")
            values.append(value)
    if len(values) != 81:
        raise ValueError(f"candidate grid does not contain 81 cells for {row.get('id')}")
    for cell, value in enumerate(board):
        if value != "0" and values[cell] != value:
            raise ValueError(f"filled cell disagrees with candidates for {row.get('id')}")
    return board, values


def coordinate(cell: int) -> str:
    row, column = divmod(cell, 9)
    return f"r{row + 1}c{column + 1}"


def unit_cells(kind: str, index: int) -> list[int]:
    if kind == "row":
        return [index * 9 + column for column in range(9)]
    if kind == "column":
        return [row * 9 + index for row in range(9)]
    if kind == "box":
        first_row, first_column = (index // 3) * 3, (index % 3) * 3
        return [
            (first_row + row) * 9 + first_column + column
            for row in range(3)
            for column in range(3)
        ]
    raise ValueError(f"unknown unit kind: {kind}")


def peers(cell: int) -> set[int]:
    row, column = divmod(cell, 9)
    box = (row // 3) * 3 + column // 3
    result = set(unit_cells("row", row)) | set(unit_cells("column", column))
    result |= set(unit_cells("box", box))
    result.discard(cell)
    return result


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


def system_prompt() -> str:
    return (
        "You are a precise Sudoku state representation engine. Use only the visible "
        "board and candidate grid. Coordinates r1c1 through r9c9 are one-based for "
        "human-readable labels. Cell indexes and unit indexes in JSON are zero-based. "
        "A box is a 3x3 region. Return exactly one JSON object and nothing else."
    )


def task_prompt(grid: str, task: str, question: str) -> str:
    return (
        "Read the Sudoku state below. Do not solve the puzzle and do not use a hidden "
        "solution.\n\n"
        f"{grid}\n\n"
        f"Task: {task}\n"
        f"Question: {question}\n"
        "Return the requested answer as JSON."
    )


def candidate_queries(
    board: str, values: list[str], preferred_kind: str | None = None
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    kinds = (preferred_kind,) if preferred_kind else ("row", "column", "box")
    for kind in kinds:
        for index in range(9):
            for digit in range(1, 10):
                cells = [
                    cell
                    for cell in unit_cells(kind, index)
                    if board[cell] == "0" and str(digit) in values[cell]
                ]
                queries.append(
                    {
                        "kind": kind,
                        "index": index,
                        "digit": digit,
                        "cells": cells,
                    }
                )
    return queries


def choose_candidate_query(
    board: str,
    values: list[str],
    state_index: int,
    want_nonempty: bool,
) -> dict[str, Any]:
    preferred_kind = ("row", "column", "box")[state_index % 3]
    preferred = candidate_queries(board, values, preferred_kind)
    matching = [query for query in preferred if bool(query["cells"]) == want_nonempty]
    if not matching:
        matching = [
            query
            for query in candidate_queries(board, values)
            if bool(query["cells"]) == want_nonempty
        ]
    if not matching:
        raise ValueError(
            f"state {state_index} has no {'nonempty' if want_nonempty else 'empty'} candidate query"
        )
    return matching[state_index % len(matching)]


def choose_peer_pair(state_index: int) -> tuple[int, int, str]:
    relation = PEER_RELATIONS[state_index % len(PEER_RELATIONS)]
    anchor = (state_index * 7) % 81
    row, column = divmod(anchor, 9)
    if relation == "same_row":
        other_box_column = (column // 3 + 1) % 3
        other_column = other_box_column * 3 + column % 3
        return anchor, row * 9 + other_column, relation
    if relation == "same_column":
        other_box_row = (row // 3 + 1) % 3
        other_row = other_box_row * 3 + row % 3
        return anchor, other_row * 9 + column, relation
    if relation == "same_box":
        box_row, box_column = row // 3, column // 3
        local = (row % 3) * 3 + column % 3
        other_local = (local + 4) % 9
        return (
            anchor,
            (box_row * 3 + other_local // 3) * 9
            + box_column * 3
            + other_local % 3,
            relation,
        )
    other = ((row + 3) % 9) * 9 + ((column + 3) % 9)
    return anchor, other, relation


def make_task(
    board: str,
    values: list[str],
    grid: str,
    task: str,
    rng: random.Random,
    state_index: int | None = None,
    task_index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if task == "cell_lookup":
        cell = (
            (state_index * 37 + task_index * 11) % 81
            if state_index is not None and task_index is not None
            else rng.randrange(81)
        )
        value = board[cell] if board[cell] != "0" else f"[{values[cell]}]"
        return task_prompt(grid, task, f"What is the visible value or candidate set at {coordinate(cell)}?"), {
            "task": task,
            "coordinate": coordinate(cell),
            "value": value,
        }

    if task == "coordinate_lookup":
        if state_index is not None and task_index is not None:
            if task_index % len(V2_TASK_PLAN) == 6:
                boundary = [
                    cell
                    for cell in range(81)
                    if divmod(cell, 9)[0] % 3 in (0, 2)
                    or divmod(cell, 9)[1] % 3 in (0, 2)
                ]
                cell = boundary[state_index % len(boundary)]
            else:
                cell = state_index % 81
        else:
            cell = rng.randrange(81)
        row, column = divmod(cell, 9)
        box = (row // 3) * 3 + column // 3
        return task_prompt(grid, task, f"Give the zero-based cell, row, column, and box indexes for {coordinate(cell)}."), {
            "task": task,
            "coordinate": coordinate(cell),
            "cell": cell,
            "row": row,
            "column": column,
            "box": box,
        }

    if task == "unit_cells":
        if state_index is not None:
            kind = ("row", "column", "box")[state_index % 3]
            index = (state_index // 3) % 9
        else:
            kind = rng.choice(("row", "column", "box"))
            index = rng.randrange(9)
        cells = unit_cells(kind, index)
        return task_prompt(grid, task, f"List the nine coordinates in zero-based {kind} {index}."), {
            "task": task,
            "unit": {"kind": kind, "index": index},
            "cells": [coordinate(cell) for cell in cells],
        }

    if task == "candidate_locations":
        if state_index is not None:
            want_nonempty = (task_index or 0) % len(V2_TASK_PLAN) == 3
            query = choose_candidate_query(board, values, state_index, want_nonempty)
            kind, index, digit, cells = (
                query["kind"],
                query["index"],
                query["digit"],
                query["cells"],
            )
        else:
            kind = rng.choice(("row", "column", "box"))
            index = rng.randrange(9)
            digit = rng.randrange(1, 10)
            cells = [
                cell
                for cell in unit_cells(kind, index)
                if board[cell] == "0" and str(digit) in values[cell]
            ]
        return task_prompt(
            grid,
            task,
            f"List the empty-cell coordinates in zero-based {kind} {index} whose candidate set contains digit {digit}.",
        ), {
            "task": task,
            "unit": {"kind": kind, "index": index},
            "digit": digit,
            "cells": [coordinate(cell) for cell in cells],
        }

    if task == "peer_relation":
        if state_index is not None:
            first, second, _ = choose_peer_pair(state_index)
        else:
            first, second = rng.sample(range(81), 2)
        first_row, first_column = divmod(first, 9)
        second_row, second_column = divmod(second, 9)
        same_row = first_row == second_row
        same_column = first_column == second_column
        same_box = (first_row // 3, first_column // 3) == (
            second_row // 3,
            second_column // 3,
        )
        return task_prompt(
            grid,
            task,
            f"Are {coordinate(first)} and {coordinate(second)} peers? Report whether they share a row, column, or box.",
        ), {
            "task": task,
            "first": coordinate(first),
            "second": coordinate(second),
            "same_row": same_row,
            "same_column": same_column,
            "same_box": same_box,
            "peers": same_row or same_column or same_box,
        }

    if task == "naked_single_scan":
        singles = [
            {"coordinate": coordinate(cell), "digit": int(values[cell])}
            for cell in range(81)
            if board[cell] == "0" and len(values[cell]) == 1
        ]
        return task_prompt(grid, task, "List every empty cell with exactly one candidate."), {
            "task": task,
            "singles": singles,
        }

    raise ValueError(f"unknown task: {task}")


def build_examples(
    rows: list[dict[str, Any]],
    tasks_per_state: int,
    seed: int,
    format_version: int = SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    if tasks_per_state <= 0:
        raise ValueError("tasks per state must be positive")
    states: dict[tuple[str, str, tuple[str, ...]], tuple[str, str, list[str]]] = {}
    for row in rows:
        board, values = parse_state(row)
        key = (row["puzzle"], board, tuple(values))
        states.setdefault(key, (row["puzzle"], row["split"], values))

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for state_index, (key, (puzzle, split, values)) in enumerate(sorted(states.items())):
        _, board, _ = key
        grid = render_grid(board, values)
        for task_index in range(tasks_per_state):
            if format_version == 1:
                task = TASKS[task_index % len(TASKS)]
                prompt, answer = make_task(board, values, grid, task, rng)
            else:
                task = V2_TASK_PLAN[task_index % len(V2_TASK_PLAN)]
                prompt, answer = make_task(
                    board,
                    values,
                    grid,
                    task,
                    rng,
                    state_index=state_index,
                    task_index=task_index,
                )
            metadata: dict[str, Any] = {}
            unit = answer.get("unit")
            if isinstance(unit, dict):
                metadata["unit_kind"] = unit.get("kind")
                metadata["unit_index"] = unit.get("index")
            if task == "candidate_locations":
                count = len(answer["cells"])
                metadata["expected_candidate_count"] = count
                metadata["candidate_stratum"] = (
                    "empty"
                    if count == 0
                    else "one"
                    if count == 1
                    else "few"
                    if count <= 3
                    else "many"
                )
            if task == "peer_relation":
                if answer["same_row"]:
                    metadata["peer_relation"] = "same_row"
                elif answer["same_column"]:
                    metadata["peer_relation"] = "same_column"
                elif answer["same_box"]:
                    metadata["peer_relation"] = "same_box"
                else:
                    metadata["peer_relation"] = "disjoint"
            identity = (
                f"v{format_version}|{puzzle}|{board}|{','.join(values)}|"
                f"{task_index}|{seed}"
            )
            examples.append(
                {
                    "schema": "sudoku-representation-example",
                    "schema_version": format_version,
                    "id": f"representation-{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
                    "source_puzzle": puzzle,
                    "source_state_index": state_index,
                    "split": split,
                    "task": task,
                    "system_prompt": system_prompt(),
                    "prompt": prompt,
                    "completion": json.dumps(answer, sort_keys=True, separators=(",", ":")),
                    "state": {"board": board, "grid": grid},
                    **metadata,
                }
            )
    return examples


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    examples = build_examples(
        rows, args.tasks_per_state, args.seed, args.format_version
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, sort_keys=True) + "\n")

    report = {
        "schema": "sudoku-representation-dataset-report",
        "schema_version": args.format_version,
        "input_files": [str(path) for path in args.input],
        "output": str(args.output),
        "examples": len(examples),
        "states": len({example["source_state_index"] for example in examples}),
        "tasks_per_state": args.tasks_per_state,
        "seed": args.seed,
        "counts_by_split": dict(Counter(example["split"] for example in examples)),
        "counts_by_task": dict(Counter(example["task"] for example in examples)),
        "counts_by_unit_kind": dict(
            Counter(
                example["unit_kind"]
                for example in examples
                if example.get("unit_kind") is not None
            )
        ),
        "counts_by_candidate_stratum": dict(
            Counter(
                example["candidate_stratum"]
                for example in examples
                if example.get("candidate_stratum") is not None
            )
        ),
        "solution_exposure": "solution omitted",
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
