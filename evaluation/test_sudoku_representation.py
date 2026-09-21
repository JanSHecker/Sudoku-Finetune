import json
import random
import unittest

from build_sudoku_deduction_dataset import candidate_grid_lines
from build_sudoku_representation_dataset import (
    build_examples,
    coordinate,
    make_task,
    peers,
    render_grid,
    unit_cells,
)


class SudokuRepresentationTests(unittest.TestCase):
    def test_coordinate_and_units_use_expected_indexes(self):
        self.assertEqual(coordinate(0), "r1c1")
        self.assertEqual(coordinate(74), "r9c3")
        self.assertEqual(unit_cells("box", 4), [30, 31, 32, 39, 40, 41, 48, 49, 50])
        self.assertIn(40, peers(30))
        self.assertNotIn(30, peers(30))

    def test_render_grid_marks_box_boundaries_and_coordinates(self):
        board = "0" * 81
        values = ["123456789"] * 81
        values[74] = "4"
        grid = render_grid(board, values)
        self.assertIn("c1 c2 c3 | c4 c5 c6 | c7 c8 c9", grid)
        self.assertIn("r9:  [123456789] [123456789] [4] |", grid)
        self.assertIn("---------+---------+---------", grid)

    def test_builder_deduplicates_states_and_emits_all_task_types(self):
        board = "0" * 81
        values = ["123456789"] * 81
        values[0] = "1"
        values[40] = "9"
        row = {
            "schema": "sudoku-deduction-example",
            "schema_version": 1,
            "id": "source-1",
            "puzzle": board,
            "split": "train",
            "state": {
                "board": board,
                "candidate_grid": candidate_grid_lines(board, values),
            },
        }
        examples = build_examples([row, dict(row, id="source-2")], 6, 42)
        self.assertEqual(len(examples), 6)
        self.assertEqual({example["task"] for example in examples}, {
            "cell_lookup",
            "coordinate_lookup",
            "unit_cells",
            "candidate_locations",
            "peer_relation",
            "naked_single_scan",
        })
        self.assertTrue(all(example["split"] == "train" for example in examples))
        self.assertTrue(all("target_deduction" not in example for example in examples))

    def test_coordinate_lookup_covers_the_box_formula(self):
        board = "0" * 81
        values = ["123456789"] * 81
        for cell in range(81):
            _, answer = make_task(
                board,
                values,
                "grid",
                "coordinate_lookup",
                random.Random(42),
                state_index=cell,
                task_index=1,
            )
            row, column = divmod(cell, 9)
            self.assertEqual(answer["cell"], cell)
            self.assertEqual(answer["row"], row)
            self.assertEqual(answer["column"], column)
            self.assertEqual(answer["box"], (row // 3) * 3 + column // 3)

    def test_v2_candidate_queries_include_empty_and_nonempty_cases(self):
        board = "123456789" + "0" * 72
        values = list("123456789") + ["123456789"] * 72
        row = {
            "schema": "sudoku-deduction-example",
            "schema_version": 1,
            "id": "source-1",
            "puzzle": board,
            "split": "train",
            "state": {
                "board": board,
                "candidate_grid": candidate_grid_lines(board, values),
            },
        }
        examples = build_examples([row], 8, 42)
        candidate_examples = [
            json.loads(example["completion"])
            for example in examples
            if example["task"] == "candidate_locations"
        ]
        self.assertEqual(len(candidate_examples), 2)
        self.assertEqual(
            {len(answer["cells"]) == 0 for answer in candidate_examples},
            {True, False},
        )


if __name__ == "__main__":
    unittest.main()
