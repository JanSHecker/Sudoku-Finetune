import unittest
import json

from build_sudoku_deduction_dataset import apply_deduction, candidate_grid_lines
from evaluate_sudoku_deductions import score_prediction
from validate_sudoku_deduction_dataset import initial_candidates, valid_solution


class SudokuDeductionDatasetTests(unittest.TestCase):
    def test_candidate_grid_preserves_spatial_rows_and_marks_empty_cells(self):
        board = "0" * 81
        values = ["123456789"] * 81
        lines = candidate_grid_lines(board, values)
        self.assertEqual(len(lines), 9)
        self.assertEqual(len(lines[0].split()), 9)
        self.assertEqual(lines[0].split()[0], "[123456789]")

    def test_naked_single_transition_is_local_and_replayable(self):
        board = "0" * 81
        values = ["123456789"] * 81
        values[0] = "1"
        deduction = {
            "technique": {"kind": "naked_single"},
            "placements": [{"cell": 0, "digit": 1}],
            "eliminations": [],
            "witness": {"unit": {"unit": {"row": 0}, "cells": [0], "digits": [1]}},
        }
        after_board, after_values = apply_deduction(board, values, deduction)
        self.assertEqual(after_board[0], "1")
        self.assertNotIn("1", after_values[1])
        self.assertNotIn("1", after_values[9])

    def test_initial_candidates_retain_the_known_solution_digit(self):
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
        self.assertTrue(valid_solution(solution))
        candidates = initial_candidates(puzzle)
        for cell, value in enumerate(puzzle):
            if value == "0":
                self.assertIn(solution[cell], candidates[cell])

    def test_evaluator_accepts_any_sound_reference_deduction(self):
        board = "0" * 81
        values = ["123456789"] * 81
        values[0] = "1"
        lines = candidate_grid_lines(board, values)
        deduction = {
            "technique": {"kind": "naked_single"},
            "placements": [{"cell": 0, "digit": 1}],
            "eliminations": [],
            "witness": {"unit": {"unit": {"row": 0}, "cells": [0], "digits": [1]}},
        }
        row = {
            "state": {"board": board, "candidate_grid": lines},
            "target_rule": "naked_single",
            "target_deduction": deduction,
            "valid_deductions": [deduction],
        }
        scored = score_prediction(row, json.dumps(deduction))
        self.assertTrue(scored["parseable"])
        self.assertTrue(scored["sound_deduction"])
        self.assertTrue(scored["valid_deduction"])

    def test_evaluator_rejects_non_json_output(self):
        row = {
            "state": {"board": "0" * 81, "candidate_grid": ["[123456789] " * 8 + "[123456789]"] * 9},
            "target_rule": "naked_single",
            "target_deduction": {},
            "valid_deductions": [],
        }
        self.assertFalse(score_prediction(row, "The answer is not JSON")["parseable"])


if __name__ == "__main__":
    unittest.main()
