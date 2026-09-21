import unittest

from evaluate_sudoku_representation import (
    parse_prediction,
    same_schema,
    score_prediction,
    summarize,
)


class SudokuRepresentationEvaluationTests(unittest.TestCase):
    def test_parser_requires_raw_json_without_repair(self):
        self.assertEqual(parse_prediction('{"task":"cell_lookup"}', "Qwen/Qwen3.5-2B"), {
            "task": "cell_lookup"
        })
        self.assertIsNone(
            parse_prediction("```json\n{\"task\":\"cell_lookup\"}\n```", "Qwen/Qwen3.5-2B")
        )

    def test_schema_check_distinguishes_booleans_from_integers(self):
        expected = {"task": "peer_relation", "peers": True}
        self.assertTrue(same_schema(expected, expected))
        self.assertFalse(same_schema({"task": "peer_relation", "peers": 1}, expected))

    def test_summary_reports_overall_and_per_task_rates(self):
        records = [
            {
                "task": "cell_lookup",
                "parseable": True,
                "schema_valid": True,
                "task_correct": True,
                "exact_answer": True,
                "set_exact_answer": True,
                "generation_seconds": 1.0,
                "expected_answer": {"task": "cell_lookup", "value": "1"},
                "parsed_prediction": {"task": "cell_lookup", "value": "1"},
            },
            {
                "task": "cell_lookup",
                "parseable": True,
                "schema_valid": False,
                "task_correct": True,
                "exact_answer": False,
                "set_exact_answer": False,
                "generation_seconds": 3.0,
                "expected_answer": {"task": "cell_lookup", "value": "2"},
                "parsed_prediction": {"task": "cell_lookup", "value": "3"},
            },
        ]
        result = summarize(records)
        self.assertEqual(result["overall"]["count"], 2)
        self.assertEqual(result["overall"]["exact_answer_rate"], 0.5)
        self.assertEqual(result["by_task"]["cell_lookup"]["schema_rate"], 0.5)

    def test_candidate_summary_balances_empty_and_nonempty_cases(self):
        records = [
            {
                "task": "candidate_locations",
                "parseable": True,
                "schema_valid": True,
                "task_correct": True,
                "exact_answer": True,
                "set_exact_answer": True,
                "generation_seconds": 1.0,
                "expected_answer": {"cells": []},
                "parsed_prediction": {"cells": []},
                "candidate_true_positive": 0,
                "candidate_false_positive": 0,
            },
            {
                "task": "candidate_locations",
                "parseable": True,
                "schema_valid": True,
                "task_correct": True,
                "exact_answer": False,
                "set_exact_answer": False,
                "generation_seconds": 1.0,
                "expected_answer": {"cells": ["r1c1", "r1c2"]},
                "parsed_prediction": {"cells": []},
                "candidate_true_positive": 0,
                "candidate_false_positive": 0,
            },
        ]
        result = summarize(records)["candidate_locations"]
        self.assertEqual(result["empty_cases"], 1)
        self.assertEqual(result["nonempty_cases"], 1)
        self.assertEqual(result["balanced_set_exact_rate"], 0.5)

    def test_candidate_metadata_is_stratified(self):
        row = {
            "id": "candidate-1",
            "task": "candidate_locations",
            "completion": '{"cells":["r1c1"],"digit":1,"task":"candidate_locations","unit":{"index":0,"kind":"row"}}',
            "unit_kind": "row",
        }
        record = score_prediction(
            row,
            '{"cells":["r1c1"],"digit":1,"task":"candidate_locations","unit":{"index":0,"kind":"row"}}',
            "Qwen/Qwen3.5-2B",
            0.1,
        )
        result = summarize([record])["candidate_locations"]
        self.assertEqual(record["candidate_stratum"], "one")
        self.assertEqual(result["by_unit_kind"]["row"]["count"], 1)
        self.assertEqual(result["by_candidate_stratum"]["one"]["count"], 1)

    def test_coordinate_metadata_is_stratified_by_box(self):
        row = {
            "id": "coordinate-1",
            "task": "coordinate_lookup",
            "completion": '{"box":1,"cell":3,"column":3,"coordinate":"r1c4","row":0,"task":"coordinate_lookup"}',
        }
        record = score_prediction(
            row,
            '{"box":1,"cell":3,"column":3,"coordinate":"r1c4","row":0,"task":"coordinate_lookup"}',
            "Qwen/Qwen3.5-2B",
            0.1,
        )
        result = summarize([record])
        self.assertEqual(result["diagnostics"]["coordinate_lookup"]["by_expected_box"]["1"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
