"""Evaluate a saved Sudoku representation adapter by free generation.

The training script reports teacher-forced loss and token accuracy.  This
evaluator instead asks the adapter to generate answers for held-out examples
and scores the raw JSON output without repairing it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_sudoku import chat_template_kwargs, load_model, model_revision  # noqa: E402
from train_sudoku_representation import read_jsonl  # noqa: E402


DEFAULT_DATASET = Path("artifacts/datasets/training/representation/sudoku-representation-v1.jsonl")
DEFAULT_PREDICTIONS = Path(
    "artifacts/results/validation/representation/sudoku-representation-v1.test.predictions.jsonl"
)
DEFAULT_METRICS = Path("artifacts/results/validation/representation/sudoku-representation-v1.test.metrics.json")
MAX_INPUT_TOKENS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['id']}|{row['completion']}\n".encode())
    return digest.hexdigest()


def parse_prediction(raw_output: str, model_id: str) -> dict[str, Any] | None:
    if model_id.casefold().startswith("liquidai/lfm2") and "</think>" in raw_output:
        raw_output = raw_output.rsplit("</think>", 1)[-1]
    try:
        value = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def same_schema(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(expected)
            and all(same_schema(value[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        if not isinstance(value, list):
            return False
        return not expected or all(same_schema(item, expected[0]) for item in value)
    if isinstance(expected, bool):
        return type(value) is bool
    if isinstance(expected, int):
        return type(value) is int
    if isinstance(expected, str):
        return isinstance(value, str)
    if expected is None:
        return value is None
    return type(value) is type(expected)


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [center - margin, center + margin]


def canonical_item(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def score_prediction(
    row: dict[str, Any], raw_output: str, model_id: str, generation_seconds: float
) -> dict[str, Any]:
    expected = json.loads(row["completion"])
    parsed = parse_prediction(raw_output, model_id)
    parseable = parsed is not None
    schema_valid = parseable and same_schema(parsed, expected)
    task_correct = parseable and parsed.get("task") == row["task"]
    exact_answer = parseable and parsed == expected
    set_exact_answer = exact_answer
    unit = expected.get("unit")
    candidate_count = (
        len(expected.get("cells", []))
        if row["task"] == "candidate_locations"
        else None
    )
    candidate_stratum = None
    if candidate_count is not None:
        candidate_stratum = (
            "empty"
            if candidate_count == 0
            else "one"
            if candidate_count == 1
            else "few"
            if candidate_count <= 3
            else "many"
        )
    list_field = {
        "candidate_locations": "cells",
        "naked_single_scan": "singles",
    }.get(row["task"])
    list_true_positive = None
    list_false_positive = None
    list_false_negative = None
    if list_field is not None:
        expected_items = {
            canonical_item(item) for item in expected[list_field]
        }
        predicted_items = {
            canonical_item(item)
            for item in (
                parsed.get(list_field, [])
                if isinstance(parsed, dict)
                and isinstance(parsed.get(list_field), list)
                else []
            )
        }
        list_true_positive = len(expected_items & predicted_items)
        list_false_positive = len(predicted_items - expected_items)
        list_false_negative = len(expected_items - predicted_items)
        set_exact_answer = (
            parseable
            and schema_valid
            and parsed.get("task") == expected["task"]
            and {
                key: parsed.get(key)
                for key in parsed
                if key != list_field
            }
            == {
                key: expected.get(key)
                for key in expected
                if key != list_field
            }
            and list_false_positive == 0
            and list_false_negative == 0
        )
    return {
        "id": row["id"],
        "task": row["task"],
        "parseable": parseable,
        "schema_valid": schema_valid,
        "task_correct": task_correct,
        "exact_answer": exact_answer,
        "set_exact_answer": set_exact_answer,
        "list_field": list_field,
        "list_true_positive": list_true_positive,
        "list_false_positive": list_false_positive,
        "list_false_negative": list_false_negative,
        "candidate_true_positive": list_true_positive
        if row["task"] == "candidate_locations"
        else None,
        "candidate_false_positive": list_false_positive
        if row["task"] == "candidate_locations"
        else None,
        "candidate_false_negative": list_false_negative
        if row["task"] == "candidate_locations"
        else None,
        "generation_seconds": generation_seconds,
        "raw_output": raw_output,
        "parsed_prediction": parsed,
        "expected_answer": expected,
        "unit_kind": unit.get("kind") if isinstance(unit, dict) else row.get("unit_kind"),
        "unit_index": unit.get("index") if isinstance(unit, dict) else row.get("unit_index"),
        "expected_answer_box": expected.get("box")
        if row["task"] == "coordinate_lookup"
        else None,
        "expected_candidate_count": candidate_count,
        "candidate_stratum": candidate_stratum,
        "peer_relation": row.get("peer_relation"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def grouped(
        group: list[dict[str, Any]], field: str
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            value = record.get(field)
            if value is not None:
                groups[str(value)].append(record)
        return {key: one(value) for key, value in sorted(groups.items())}

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        exact_count = sum(item["exact_answer"] for item in group)
        expected_answers = collections.Counter(
            canonical_item(item["expected_answer"]) for item in group
        )
        return {
            "count": count,
            "parse_rate": sum(item["parseable"] for item in group) / count
            if count
            else 0.0,
            "schema_rate": sum(item["schema_valid"] for item in group) / count
            if count
            else 0.0,
            "task_rate": sum(item["task_correct"] for item in group) / count
            if count
            else 0.0,
            "exact_answer_rate": exact_count / count if count else 0.0,
            "exact_answer_wilson_95": wilson_interval(exact_count, count),
            "majority_exact_baseline": (
                max(expected_answers.values()) / count if count else 0.0
            ),
            "set_exact_answer_rate": sum(
                item["set_exact_answer"] for item in group
            )
            / count
            if count
            else 0.0,
            "mean_generation_seconds": sum(
                item["generation_seconds"] for item in group
            )
            / count
            if count
            else 0.0,
        }

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    candidate_records = by_task.get("candidate_locations", [])
    empty_cases = [
        record
        for record in candidate_records
        if record["expected_answer"]["cells"] == []
    ]
    nonempty_cases = [
        record
        for record in candidate_records
        if record["expected_answer"]["cells"]
    ]

    def rate(group: list[dict[str, Any]], field: str) -> float:
        return sum(item[field] for item in group) / len(group) if group else 0.0

    def list_summary(group: list[dict[str, Any]], field: str) -> dict[str, Any]:
        empty = [record for record in group if not record["expected_answer"][field]]
        nonempty = [record for record in group if record["expected_answer"][field]]

        def list_value(record: dict[str, Any], name: str) -> int:
            value = record.get(f"list_{name}")
            if value is None and field == "cells":
                value = record.get(f"candidate_{name}")
            return value or 0

        recalls = []
        precisions = []
        for record in nonempty:
            expected_count = len(record["expected_answer"][field])
            predicted = record["parsed_prediction"]
            predicted_count = (
                len(predicted.get(field, [])) if isinstance(predicted, dict) else 0
            )
            recalls.append(list_value(record, "true_positive") / expected_count)
            precisions.append(
                list_value(record, "true_positive") / predicted_count
                if predicted_count
                else 0.0
            )
        true_positive = sum(list_value(record, "true_positive") for record in nonempty)
        false_positive = sum(list_value(record, "false_positive") for record in nonempty)
        false_negative = sum(list_value(record, "false_negative") for record in nonempty)
        return {
            "count": len(group),
            "empty_cases": len(empty),
            "nonempty_cases": len(nonempty),
            "empty_set_exact_rate": rate(empty, "set_exact_answer"),
            "nonempty_set_exact_rate": rate(nonempty, "set_exact_answer"),
            "balanced_set_exact_rate": (
                rate(empty, "set_exact_answer")
                + rate(nonempty, "set_exact_answer")
            )
            / 2
            if empty and nonempty
            else 0.0,
            "empty_majority_baseline": len(empty) / len(group) if group else 0.0,
            "nonempty_mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "nonempty_mean_precision": (
                sum(precisions) / len(precisions) if precisions else 0.0
            ),
            "nonempty_micro_recall": true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0,
            "nonempty_micro_precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0,
        }

    def field_summary(group: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in fields:
            expected = [record["expected_answer"].get(field) for record in group]
            predicted = [
                record["parsed_prediction"].get(field)
                if isinstance(record["parsed_prediction"], dict)
                else None
                for record in group
            ]
            correct = sum(actual == guess for actual, guess in zip(expected, predicted))
            counts = collections.Counter(canonical_item(value) for value in expected)
            field_result: dict[str, Any] = {
                "count": len(group),
                "accuracy": correct / len(group) if group else 0.0,
                "wilson_95": wilson_interval(correct, len(group)),
                "majority_baseline": max(counts.values()) / len(group)
                if group
                else 0.0,
                "distinct_expected_values": len(counts),
            }
            if len(counts) <= 20:
                field_result["expected_counts"] = dict(counts)
            if all(isinstance(value, bool) for value in expected):
                positives = sum(value is True for value in expected)
                negatives = sum(value is False for value in expected)
                true_positive = sum(
                    actual is True and guess is True
                    for actual, guess in zip(expected, predicted)
                )
                true_negative = sum(
                    actual is False and guess is False
                    for actual, guess in zip(expected, predicted)
                )
                field_result["balanced_accuracy"] = (
                    (true_positive / positives if positives else 0.0)
                    + (true_negative / negatives if negatives else 0.0)
                ) / (2 if positives and negatives else 1)
            result[field] = field_result
        return result

    task_summaries = {task: one(rows) for task, rows in sorted(by_task.items())}
    metric_names = (
        "parse_rate",
        "schema_rate",
        "task_rate",
        "exact_answer_rate",
        "set_exact_answer_rate",
    )
    macro_average = {
        name: sum(summary[name] for summary in task_summaries.values())
        / len(task_summaries)
        if task_summaries
        else 0.0
        for name in metric_names
    }
    candidate_summary = list_summary(candidate_records, "cells")
    candidate_summary["by_unit_kind"] = grouped(candidate_records, "unit_kind")
    candidate_summary["by_candidate_stratum"] = grouped(
        candidate_records, "candidate_stratum"
    )
    naked_summary = list_summary(by_task.get("naked_single_scan", []), "singles")
    return {
        "overall": one(records),
        "task_macro_average": macro_average,
        "task_distribution": {task: len(rows) for task, rows in sorted(by_task.items())},
        "by_task": task_summaries,
        "candidate_locations": candidate_summary,
        "diagnostics": {
            "candidate_locations": candidate_summary,
            "naked_single_scan": naked_summary,
            "peer_relation": {
                "fields": field_summary(
                    by_task.get("peer_relation", []),
                    ["same_row", "same_column", "same_box", "peers"],
                )
            },
            "coordinate_lookup": {
                "fields": field_summary(
                    by_task.get("coordinate_lookup", []),
                    ["cell", "row", "column", "box"],
                ),
                "by_expected_box": grouped(
                    by_task.get("coordinate_lookup", []), "expected_answer_box"
                ),
            },
            "cell_lookup": {
                "fields": field_summary(by_task.get("cell_lookup", []), ["value"])
            },
            "unit_cells": {
                "fields": field_summary(by_task.get("unit_cells", []), ["cells"])
            },
        },
    }


def load_existing(path: Path, dataset_hash: str) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("dataset_sha256") != dataset_hash:
                raise ValueError(
                    f"prediction file has a different dataset at line {line_number}"
                )
            existing[record["id"]] = record
    return existing


def generate_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    model_id: str,
) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(model_id),
        )
        for row in rows
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise ValueError("representation prompt exceeded the input token budget")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--batch-size and --max-new-tokens must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if not args.adapter.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {args.adapter}")

    all_rows = read_jsonl([args.dataset])
    rows = [row for row in all_rows if row["split"] == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no rows found for split {args.split}")
    dataset_hash = dataset_fingerprint(rows)

    if args.predictions.exists() and not args.resume and not args.overwrite:
        raise SystemExit("prediction file exists; use --resume or --overwrite")
    if args.overwrite and args.predictions.exists():
        args.predictions.unlink()
    existing = load_existing(args.predictions, dataset_hash) if args.resume else {}
    for row in rows:
        record = existing.get(row["id"])
        if record is not None and isinstance(record.get("raw_output"), str):
            record.update(
                score_prediction(
                    row,
                    record["raw_output"],
                    args.model_id,
                    float(record.get("generation_seconds", 0.0)),
                )
            )
    pending = [row for row in rows if row["id"] not in existing]

    started = time.perf_counter()
    model = None
    if pending:
        revision = model_revision(args.model_id, args.revision)
        model, tokenizer = load_model(args.model_id, revision)
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()
        tokenizer.padding_side = "left"
        torch.cuda.reset_peak_memory_stats()
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                generation_started = time.perf_counter()
                outputs = generate_batch(
                    model, tokenizer, batch, args.max_new_tokens, args.model_id
                )
                batch_seconds = time.perf_counter() - generation_started
                per_example_seconds = batch_seconds / len(batch)
                for row, raw_output in zip(batch, outputs):
                    record = {
                        "dataset_sha256": dataset_hash,
                        "model_id": args.model_id,
                        "model_revision": revision,
                        "adapter": str(args.adapter),
                        "split": args.split,
                        "generation": {
                            "do_sample": False,
                            "max_new_tokens": args.max_new_tokens,
                        },
                        **score_prediction(
                            row, raw_output, args.model_id, per_example_seconds
                        ),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    existing[row["id"]] = record
                stream.flush()
                print(
                    f"evaluated {min(start + args.batch_size, len(pending))}/"
                    f"{len(pending)}",
                    flush=True,
                )

    ordered = [existing[row["id"]] for row in rows]
    revision = model_revision(args.model_id, args.revision)
    metrics = {
        "schema": "sudoku-representation-generation-metrics-v2",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "predictions": str(args.predictions),
        "split": args.split,
        "examples": len(rows),
        "model_id": args.model_id,
        "model_revision": revision,
        "adapter": str(args.adapter),
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        **summarize(ordered),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if model is not None else None
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
