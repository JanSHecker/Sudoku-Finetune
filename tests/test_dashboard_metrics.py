import json
import tempfile
import unittest
from pathlib import Path

from dashboard.catalog import discover_runs, summarize_run
from dashboard.metrics import classification_metrics, list_metrics


class DashboardMetricTests(unittest.TestCase):
    def test_macro_metrics_expose_majority_class_failure(self):
        records = [
            {"target": "common", "prediction": "common"}
            for _ in range(95)
        ] + [
            {"target": "rare", "prediction": "common"}
            for _ in range(5)
        ]

        result = classification_metrics(records, "target", "prediction")

        self.assertEqual(result["accuracy"], 0.95)
        self.assertEqual(result["support"], {"common": 95, "rare": 5})
        self.assertEqual(result["majority_baseline"], 0.95)
        self.assertEqual(result["macro"]["recall"], 0.5)
        self.assertLess(result["macro"]["f1"], 0.5)
        self.assertEqual(result["per_class"]["rare"]["recall"], 0.0)

    def test_unparsed_outputs_are_visible_without_becoming_a_target_class(self):
        result = classification_metrics(
            [
                {"target": "a", "prediction": "a"},
                {"target": "b", "prediction": None},
            ],
            "target",
            "prediction",
        )

        self.assertEqual(result["labels"], ["a", "b"])
        self.assertIn("__unparsed__", result["matrix_labels"])
        self.assertEqual(result["confusion_matrix"]["b"]["__unparsed__"], 1)
        self.assertEqual(result["parse_rate"], 0.5)

    def test_list_metrics_balance_empty_and_nonempty_cases(self):
        records = [
            {
                "expected_answer": {"cells": []},
                "parsed_prediction": {"cells": []},
            },
            {
                "expected_answer": {"cells": ["r1c1", "r1c2"]},
                "parsed_prediction": {"cells": ["r1c1"]},
            },
        ]

        result = list_metrics(records, "cells", "cells")

        self.assertEqual(result["empty_cases"], 1)
        self.assertEqual(result["nonempty_cases"], 1)
        self.assertEqual(result["balanced_exact_rate"], 0.5)
        self.assertEqual(result["macro"]["recall"], 0.5)

    def test_catalog_uses_explicit_prediction_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            root.mkdir()
            metrics_path = root / "custom.metrics.json"
            predictions_path = root / "custom-output.jsonl"
            predictions_path.write_text(
                json.dumps(
                    {
                        "target_rule": "single",
                        "predicted_rule": "single",
                        "parseable": True,
                        "sound_deduction": True,
                        "valid_deduction": True,
                        "valid_rule": True,
                        "exact_target": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema": "sudoku-deduction-metrics-v1",
                        "model_id": "test/model",
                        "predictions": str(predictions_path),
                    }
                ),
                encoding="utf-8",
            )

            runs = discover_runs(root)
            summary = summarize_run(runs[0])

            self.assertEqual(runs[0].predictions_path, predictions_path)
            self.assertEqual(summary["derived"]["classification"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
