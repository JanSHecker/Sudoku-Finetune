"""Build a targeted candidate-location representation augmentation.

The source is the frozen v2 representation dataset.  Only its train and
validation states are used; test states are deliberately excluded.  Queries
are sampled from every row, column, and box/digit combination and stratified by
the number of matching cells.  Non-empty box queries receive extra weight
because they are the main remaining failure mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_sudoku_representation_dataset import (
    candidate_queries,
    coordinate,
    system_prompt,
    task_prompt,
)


TASK = "candidate_locations"
KINDS = ("row", "column", "box")
STRATA = ("empty", "one", "few", "many")
SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--queries-per-state", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=("train", "validation"),
        help="Output only this split; repeat to include multiple splits. Defaults to both.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("schema") != "sudoku-representation-example"
                or row.get("schema_version") != SCHEMA_VERSION
                or not isinstance(row.get("source_puzzle"), str)
                or len(row["source_puzzle"]) != 81
                or row.get("split") not in {"train", "validation", "test"}
                or not isinstance(row.get("state"), dict)
                or not isinstance(row["state"].get("board"), str)
                or not isinstance(row["state"].get("grid"), str)
            ):
                raise ValueError(f"invalid representation row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def parse_state(row: dict[str, Any]) -> tuple[str, list[str]]:
    board = row["state"]["board"]
    if len(board) != 81 or any(character not in "0123456789" for character in board):
        raise ValueError(f"invalid board for {row.get('id')}")

    values: list[str] = []
    for line in row["state"]["grid"].splitlines():
        if not line.startswith("r"):
            continue
        tokens = [token for token in line.split()[1:] if token != "|"]
        if len(tokens) != 9:
            raise ValueError(f"invalid grid row for {row.get('id')}")
        values.extend(
            token[1:-1] if token.startswith("[") and token.endswith("]") else token
            for token in tokens
        )
    if len(values) != 81:
        raise ValueError(f"invalid candidate grid for {row.get('id')}")
    return board, values


def stratum(count: int) -> str:
    if count == 0:
        return "empty"
    if count == 1:
        return "one"
    if count <= 3:
        return "few"
    return "many"


def category_weight(kind: str, result_stratum: str) -> int:
    # Keep all empty and row/column non-empty categories represented, while
    # deliberately giving non-empty box queries twice the sampling weight.
    return 2 if kind == "box" and result_stratum != "empty" else 1


def category_targets(total: int) -> dict[tuple[str, str], int]:
    categories = [(kind, result_stratum) for result_stratum in STRATA for kind in KINDS]
    weights = {category: category_weight(*category) for category in categories}
    total_weight = sum(weights.values())
    raw = {category: total * weight / total_weight for category, weight in weights.items()}
    targets = {category: int(value) for category, value in raw.items()}
    remainder = total - sum(targets.values())
    for category in sorted(categories, key=lambda item: raw[item] - targets[item], reverse=True)[:remainder]:
        targets[category] += 1
    return targets


def state_records(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    states: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        board, values = parse_state(row)
        key = (row["source_puzzle"], board, tuple(values))
        states.setdefault(
            key,
            {
                "source_puzzle": row["source_puzzle"],
                "board": board,
                "values": values,
                "grid": row["state"]["grid"],
                "split": row["split"],
            },
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in sorted(states.values(), key=lambda item: (item["split"], item["source_puzzle"], item["board"])):
        grouped[state["split"]].append(state)
    return grouped


def candidates_for_state(state: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for query in candidate_queries(state["board"], state["values"]):
        result_stratum = stratum(len(query["cells"]))
        buckets[(query["kind"], result_stratum)].append(query)
    return buckets


def choose_rows(
    states: list[dict[str, Any]], queries_per_state: int, seed: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    target_total = len(states) * queries_per_state
    targets = category_targets(target_total)
    by_category: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for state_index, state in enumerate(states):
        for category, queries in candidates_for_state(state).items():
            shuffled = queries[:]
            random.Random(seed + state_index * 1009 + sum(map(ord, category[0]))).shuffle(shuffled)
            by_category[category].extend((state_index, query) for query in shuffled)

    chosen: list[dict[str, Any]] = []
    chosen_counts: Counter[tuple[str, str]] = Counter()
    for category in sorted(targets):
        candidates = by_category[category]
        target = targets[category]
        if len(candidates) < target:
            raise ValueError(
                f"not enough {category[0]} {category[1]} queries: "
                f"{len(candidates)} available, {target} required"
            )

        # Round-robin over states prevents the augmentation from concentrating
        # on a small number of puzzles when a category has many candidates.
        by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for state_index, query in candidates:
            by_state[state_index].append(query)
        state_order = list(sorted(by_state))
        random.Random(
            seed + len(category) * 997 + sum(map(ord, category[0]))
        ).shuffle(state_order)
        for offset in range(target):
            state_index = state_order[offset % len(state_order)]
            query = by_state[state_index].pop()
            state = states[state_index]
            chosen.append({"state": state, "query": query})
            chosen_counts[category] += 1
            if not by_state[state_index]:
                state_order.remove(state_index)
                if not state_order and offset + 1 < target:
                    raise ValueError(f"category {category} was exhausted unexpectedly")
    return chosen, dict(chosen_counts)


def make_row(item: dict[str, Any], state_index: int) -> dict[str, Any]:
    state = item["state"]
    query = item["query"]
    result_stratum = stratum(len(query["cells"]))
    answer = {
        "task": TASK,
        "unit": {"kind": query["kind"], "index": query["index"]},
        "digit": query["digit"],
        "cells": [coordinate(cell) for cell in query["cells"]],
    }
    identity = (
        f"candidate-location-v1|{state['source_puzzle']}|{state['board']}|"
        f"{','.join(state['values'])}|{query['kind']}|{query['index']}|{query['digit']}"
    )
    return {
        "schema": "sudoku-representation-example",
        "schema_version": SCHEMA_VERSION,
        "id": f"representation-candidate-{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
        "source_puzzle": state["source_puzzle"],
        "source_state_index": state_index,
        "split": state["split"],
        "task": TASK,
        "system_prompt": system_prompt(),
        "prompt": task_prompt(
            state["grid"],
            TASK,
            "List the empty-cell coordinates in zero-based "
            f"{query['kind']} {query['index']} whose candidate set contains digit {query['digit']}.",
        ),
        "completion": json.dumps(answer, sort_keys=True, separators=(",", ":")),
        "state": {"board": state["board"], "grid": state["grid"]},
        "unit_kind": query["kind"],
        "unit_index": query["index"],
        "digit": query["digit"],
        "expected_candidate_count": len(query["cells"]),
        "candidate_stratum": result_stratum,
        "augmentation": "candidate-location-v1",
    }


def write_split(
    states: list[dict[str, Any]], queries_per_state: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen, selected_counts = choose_rows(states, queries_per_state, seed)
    output_rows = [make_row(item, index) for index, item in enumerate(chosen)]
    return output_rows, {
        "states": len(states),
        "rows": len(output_rows),
        "selected_by_category": {
            f"{kind}_{result_stratum}": count
            for (kind, result_stratum), count in sorted(selected_counts.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.queries_per_state <= 0:
        raise ValueError("--queries-per-state must be positive")
    rows = read_rows(args.input)
    grouped = state_records(rows)
    selected_splits = tuple(args.splits or ("train", "validation"))
    if "train" not in selected_splits or not grouped.get("train"):
        raise ValueError("output must include non-empty train states")
    if "validation" in selected_splits and not grouped.get("validation"):
        raise ValueError("input does not contain validation states")

    output_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for split in selected_splits:
        split_rows, report = write_split(grouped[split], args.queries_per_state, args.seed)
        output_rows.extend(split_rows)
        split_reports[split] = report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    puzzles_by_split = {
        split: len({state["source_puzzle"] for state in grouped[split]})
        for split in selected_splits
    }
    report = {
        "schema": "sudoku-candidate-location-augmentation-report",
        "schema_version": 1,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "queries_per_state": args.queries_per_state,
        "output_splits": list(selected_splits),
        "source_states": {
            split: len(grouped.get(split, [])) for split in ("train", "validation", "test")
        },
        "source_puzzles": {
            split: len({state["source_puzzle"] for state in grouped.get(split, [])})
            for split in ("train", "validation", "test")
        },
        "test_states_excluded": len(grouped.get("test", [])),
        "output_rows": len(output_rows),
        "output_puzzles": puzzles_by_split,
        "by_split": split_reports,
        "target_weights": {
            f"{kind}_{result_stratum}": category_weight(kind, result_stratum)
            for result_stratum in STRATA
            for kind in KINDS
        },
        "test_leakage": "test states were excluded from output",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
