"""Validate Sudoku deduction traces and state-level examples.

This validator checks the generated state transitions independently of the Rust
serializer: candidate masks, placements, eliminations, witnesses, solution
compatibility, and trace continuity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_sudoku_deduction_dataset import apply_deduction, parse_candidate_grid


ALL_DIGITS = set("123456789")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--trace", action="store_true", help="input is raw Rust trace JSONL")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def unit_cells(unit: dict[str, Any]) -> list[int]:
    if len(unit) != 1:
        raise ValueError(f"invalid unit: {unit}")
    kind, index = next(iter(unit.items()))
    if kind == "row" and isinstance(index, int) and 0 <= index < 9:
        return [index * 9 + column for column in range(9)]
    if kind == "column" and isinstance(index, int) and 0 <= index < 9:
        return [row * 9 + index for row in range(9)]
    if kind == "box" and isinstance(index, int) and 0 <= index < 9:
        first_row, first_column = (index // 3) * 3, (index % 3) * 3
        return [
            (first_row + row) * 9 + first_column + column
            for row in range(3)
            for column in range(3)
        ]
    raise ValueError(f"invalid unit: {unit}")


def peer_set(cell: int) -> set[int]:
    row, column = divmod(cell, 9)
    box = (row // 3) * 3 + column // 3
    return (
        set(row * 9 + offset for offset in range(9))
        | set(offset * 9 + column for offset in range(9))
        | set(
            (box // 3 * 3 + local_row) * 9
            + box % 3 * 3
            + local_column
            for local_row in range(3)
            for local_column in range(3)
        )
    ) - {cell}


def valid_solution(solution: str) -> bool:
    if len(solution) != 81 or set(solution) != ALL_DIGITS:
        return False
    for row in range(9):
        if set(solution[row * 9 : row * 9 + 9]) != ALL_DIGITS:
            return False
    for column in range(9):
        if {solution[row * 9 + column] for row in range(9)} != ALL_DIGITS:
            return False
    for box in range(9):
        cells = unit_cells({"box": box})
        if {solution[cell] for cell in cells} != ALL_DIGITS:
            return False
    return True


def initial_candidates(board: str) -> list[str]:
    values = list(board)
    result: list[str] = []
    for cell, value in enumerate(values):
        if value != "0":
            result.append(value)
            continue
        unavailable = {values[peer] for peer in peer_set(cell) if values[peer] != "0"}
        result.append("".join(sorted(ALL_DIGITS - unavailable)))
    return result


def rule_key(deduction: dict[str, Any]) -> str:
    technique = deduction["technique"]
    if technique["kind"] != "fish":
        return technique["kind"]
    return f"fish_{technique['size']}_{technique['orientation']}"


def validate_witness(
    board: str, candidate_values: list[str], deduction: dict[str, Any]
) -> None:
    candidates = parse_candidate_grid(list(board), candidate_values)
    technique = deduction["technique"]
    kind = technique["kind"]
    witness = deduction["witness"]
    placements = deduction.get("placements", [])
    eliminations = deduction.get("eliminations", [])

    if kind == "naked_single":
        if len(placements) != 1 or eliminations or len(candidates[placements[0]["cell"]]) != 1:
            raise ValueError("invalid naked single witness")
        return

    if kind == "hidden_single":
        unit_info = witness.get("unit", {}).get("unit")
        cells = witness.get("unit", {}).get("cells")
        digits = witness.get("unit", {}).get("digits")
        if not isinstance(unit_info, dict) or cells is None or digits is None:
            raise ValueError("invalid hidden single witness")
        unit = unit_cells(unit_info)
        if len(placements) != 1 or len(digits) != 1 or cells != [placements[0]["cell"]]:
            raise ValueError("hidden single witness does not match placement")
        digit = str(digits[0])
        matching = [cell for cell in unit if digit in candidates[cell]]
        if matching != cells or placements[0]["digit"] != int(digit):
            raise ValueError("hidden single is not unique in its unit")
        return

    if kind in {
        "naked_pair",
        "naked_triple",
        "naked_quad",
        "hidden_pair",
        "hidden_triple",
        "hidden_quad",
    }:
        info = witness.get("unit", {})
        unit_info = info.get("unit")
        cells = info.get("cells")
        digits = {str(digit) for digit in info.get("digits", [])}
        if not isinstance(unit_info, dict) or not isinstance(cells, list):
            raise ValueError("invalid subset witness")
        if len(cells) != len(digits) or len(cells) not in (2, 3, 4):
            raise ValueError("subset size is invalid")
        unit = unit_cells(unit_info)
        if kind.startswith("naked"):
            union = set().union(*(candidates[cell] for cell in cells))
            if union != digits or any(candidates[cell] - digits for cell in cells):
                raise ValueError("invalid naked subset")
            allowed = set(cells)
            expected = {
                (cell, int(digit))
                for cell in unit
                if cell not in allowed
                for digit in candidates[cell] & digits
            }
        else:
            selected = [
                cell for cell in unit if candidates[cell] & digits
            ]
            if set(selected) != set(cells) or any(
                digit not in set().union(*(candidates[cell] for cell in cells))
                for digit in digits
            ):
                raise ValueError("invalid hidden subset")
            expected = {
                (cell, int(digit))
                for cell in cells
                for digit in candidates[cell] - digits
            }
        actual = {(item["cell"], item["digit"]) for item in eliminations}
        if actual != expected:
            raise ValueError("subset eliminations do not match witness")
        return

    if kind in {"pointing", "claiming"}:
        info = witness.get("locked", {})
        source_info, target_info = info.get("source"), info.get("target")
        digit = str(info.get("digit"))
        cells = info.get("cells")
        if not isinstance(source_info, dict) or not isinstance(target_info, dict):
            raise ValueError("invalid locked-candidate witness")
        source, target = unit_cells(source_info), unit_cells(target_info)
        matching = [cell for cell in source if digit in candidates[cell]]
        if len(matching) < 2 or set(matching) != set(cells):
            raise ValueError("locked-candidate source is invalid")
        if kind == "pointing":
            if not isinstance(source_info.get("box"), int) or not all(
                cell // 9 == matching[0] // 9 for cell in matching
            ) and not all(cell % 9 == matching[0] % 9 for cell in matching):
                raise ValueError("pointing witness is not line-locked")
            expected_cells = {
                cell for cell in target if cell not in source and digit in candidates[cell]
            }
        else:
            if source_info.get("row") is None and source_info.get("column") is None:
                raise ValueError("claiming witness is not line-based")
            if not all(
                (cell // 9 == matching[0] // 9)
                if source_info.get("row") is not None
                else (cell % 9 == matching[0] % 9)
                for cell in matching
            ):
                raise ValueError("claiming witness is not line-locked")
            expected_cells = {
                cell for cell in target if cell not in source and digit in candidates[cell]
            }
        actual = {(item["cell"], item["digit"]) for item in eliminations}
        if actual != {(cell, int(digit)) for cell in expected_cells}:
            raise ValueError("locked-candidate eliminations do not match witness")
        return

    if kind == "fish":
        info = witness.get("fish", {})
        digit = str(info.get("digit"))
        bases = info.get("base_units")
        covers = info.get("cover_units")
        size = technique.get("size")
        orientation = technique.get("orientation")
        if not isinstance(bases, list) or not isinstance(covers, list):
            raise ValueError("invalid fish witness")
        if len(bases) != size or len(covers) != size or len(set(bases)) != size:
            raise ValueError("fish size does not match witness")
        positions: list[set[int]] = []
        for base in bases:
            if not isinstance(base, int) or not 0 <= base < 9:
                raise ValueError("fish base is out of range")
            cells = (
                [base * 9 + column for column in range(9)]
                if orientation == "row"
                else [row * 9 + base for row in range(9)]
            )
            position = {
                (cell % 9 if orientation == "row" else cell // 9)
                for cell in cells
                if digit in candidates[cell]
            }
            if not 2 <= len(position) <= size:
                raise ValueError("fish base has invalid candidate count")
            positions.append(position)
        cover_set = set(covers)
        if len(cover_set) != size or set.union(*positions) != cover_set:
            raise ValueError("fish covers do not match base candidate positions")
        base_set = set(bases)
        expected: set[tuple[int, int]] = set()
        for other in range(9):
            if other in base_set:
                continue
            for cover in cover_set:
                cell = other * 9 + cover if orientation == "row" else cover * 9 + other
                if digit in candidates[cell]:
                    expected.add((cell, int(digit)))
        actual = {(item["cell"], item["digit"]) for item in eliminations}
        if actual != expected:
            raise ValueError("fish eliminations do not match witness")
        return

    raise ValueError(f"unsupported deduction type: {kind}")


def validate_trace(record: dict[str, Any]) -> Counter[str]:
    puzzle = record.get("puzzle")
    solution = record.get("solution")
    if not isinstance(puzzle, str) or not isinstance(solution, str):
        raise ValueError("trace lacks puzzle or solution")
    if len(puzzle) != 81 or any(value not in "0123456789" for value in puzzle):
        raise ValueError("invalid puzzle")
    if not valid_solution(solution):
        raise ValueError("invalid solution")
    if any(puzzle[index] != "0" and puzzle[index] != solution[index] for index in range(81)):
        raise ValueError("solution does not preserve clues")
    current_board = puzzle
    current_candidates = initial_candidates(puzzle)
    if record.get("candidate_grid") != current_candidates:
        raise ValueError("initial candidate grid is incorrect")
    counts: Counter[str] = Counter()
    for index, step in enumerate(record.get("steps", [])):
        if step.get("board_before") != current_board:
            raise ValueError(f"step {index} has discontinuous board")
        if step.get("candidate_grid_before") != current_candidates:
            raise ValueError(f"step {index} has discontinuous candidates")
        validate_witness(current_board, current_candidates, step)
        available = step.get("available_deductions", [])
        if step not in available and not any(
            deduction == {
                "technique": step["technique"],
                "placements": step["placements"],
                "eliminations": step["eliminations"],
                "witness": step["witness"],
            }
            for deduction in available
        ):
            raise ValueError(f"step {index} is not listed as available")
        for deduction in available:
            validate_witness(current_board, current_candidates, deduction)
            apply_deduction(current_board, current_candidates, deduction)
        next_board, next_candidates = apply_deduction(
            current_board,
            current_candidates,
            {
                "technique": step["technique"],
                "placements": step["placements"],
                "eliminations": step["eliminations"],
                "witness": step["witness"],
            },
        )
        if step.get("board_after") != next_board or step.get("candidate_grid_after") != next_candidates:
            raise ValueError(f"step {index} has incorrect after-state")
        if any(
            next_board[cell] == "0" and solution[cell] not in next_candidates[cell]
            for cell in range(81)
        ):
            raise ValueError(f"step {index} removes the known solution digit")
        counts[rule_key(step)] += 1
        current_board, current_candidates = next_board, next_candidates
    return counts


def validate_example(example: dict[str, Any]) -> str:
    if example.get("schema") != "sudoku-deduction-example":
        raise ValueError("unsupported example schema")
    state = example.get("state", {})
    next_state = example.get("next_state", {})
    board = state.get("board")
    candidate_lines = state.get("candidate_grid")
    if not isinstance(board, str) or not isinstance(candidate_lines, list) or len(candidate_lines) != 9:
        raise ValueError("invalid example state")
    values = [
        token[1:-1] if token.startswith("[") and token.endswith("]") else token
        for line in candidate_lines
        for token in line.split()
    ]
    if len(values) != 81:
        raise ValueError("candidate grid must have 81 tokens")
    parse_candidate_grid(list(board), values)
    target = example.get("target_deduction")
    if not isinstance(target, dict):
        raise ValueError("example has no target deduction")
    prompt = example.get("prompt")
    completion = example.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("example has no model prompt/completion")
    if "valid_deductions" in prompt or example.get("solution_sha256", "") in prompt:
        raise ValueError("prompt contains hidden label data")
    if json.loads(completion) != target:
        raise ValueError("completion does not equal target deduction")
    if any(line not in prompt for line in candidate_lines):
        raise ValueError("prompt does not contain the complete candidate grid")
    validate_witness(board, values, target)
    after_board, after_candidates = apply_deduction(board, values, target)
    expected_lines = candidate_grid_lines(after_board, after_candidates)
    if next_state.get("board") != after_board or next_state.get("candidate_grid") != expected_lines:
        raise ValueError("example transition is incorrect")
    if target not in example.get("valid_deductions", []):
        raise ValueError("target is not in valid deduction set")
    for deduction in example.get("valid_deductions", []):
        validate_witness(board, values, deduction)
        apply_deduction(board, values, deduction)
    if example.get("split") not in {"train", "validation", "test", "benchmark"}:
        raise ValueError("example has no valid split")
    return example["target_rule"]


def candidate_grid_lines(board: str, values: list[str]) -> list[str]:
    rendered = [
        value if board[index] != "0" else f"[{value}]"
        for index, value in enumerate(values)
    ]
    return [" ".join(rendered[index : index + 9]) for index in range(0, 81, 9)]


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    puzzle_splits: dict[str, str] = {}
    records = 0
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if args.trace:
                    counts.update(validate_trace(record))
                else:
                    counts[validate_example(record)] += 1
                    puzzle = record["puzzle"]
                    split = record["split"]
                    previous = puzzle_splits.setdefault(puzzle, split)
                    if previous != split:
                        raise ValueError(f"puzzle appears in multiple splits: {puzzle}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(f"invalid record at {args.input}:{line_number}: {error}") from error
            records += 1
    report = {
        "schema": "sudoku-deduction-validation-report",
        "schema_version": 1,
        "input": str(args.input),
        "records": records,
        "puzzles": len(puzzle_splits),
        "sound": True,
        "steps_or_examples_by_rule": dict(sorted(counts.items())),
    }
    report_path = args.report
    if report_path:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
