"""Metric calculations used by the dashboard.

The evaluators intentionally keep task-specific metrics in their own schemas.
This module provides a common, imbalance-aware view over their prediction
records without changing the evaluator's source-of-truth checks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


UNPARSED_LABEL = "__unparsed__"


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * (
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) ** 0.5 / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def classification_metrics(
    records: Iterable[dict[str, Any]],
    target_key: str,
    predicted_key: str,
    labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return confusion, per-class and aggregate metrics for a classifier.

    Missing predictions are represented as an ``__unparsed__`` column in the
    confusion matrix, but aggregate class metrics are averaged only over
    classes present in the target labels. This prevents parse failures from
    becoming a fake task class while still making them visible.
    """

    rows = list(records)
    targets = [str(row.get(target_key)) for row in rows if row.get(target_key) is not None]
    target_labels = list(dict.fromkeys(str(label) for label in labels or targets))
    predicted_labels = [
        str(row[predicted_key])
        if row.get(predicted_key) is not None
        else UNPARSED_LABEL
        for row in rows
    ]
    matrix_labels = list(dict.fromkeys(target_labels + predicted_labels))
    confusion = {label: {other: 0 for other in matrix_labels} for label in target_labels}
    for row in rows:
        target = row.get(target_key)
        if target is None:
            continue
        target = str(target)
        predicted = (
            str(row[predicted_key])
            if row.get(predicted_key) is not None
            else UNPARSED_LABEL
        )
        confusion.setdefault(target, {other: 0 for other in matrix_labels})
        confusion[target][predicted] = confusion[target].get(predicted, 0) + 1

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion.get(label, {}).get(label, 0) for label in target_labels)
    per_class: dict[str, dict[str, Any]] = {}
    for label in target_labels:
        true_positive = confusion.get(label, {}).get(label, 0)
        false_positive = sum(
            confusion.get(other, {}).get(label, 0)
            for other in target_labels
            if other != label
        )
        false_negative = sum(
            confusion.get(label, {}).get(other, 0)
            for other in matrix_labels
            if other != label
        )
        support = sum(confusion.get(label, {}).values())
        true_negative = total - true_positive - false_positive - false_negative
        # A class with no predicted positives still has zero precision for
        # macro scoring. Treating it as missing would hide minority failures.
        precision = _rate(true_positive, true_positive + false_positive) or 0.0
        recall = _rate(true_positive, support) or 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        specificity = _rate(true_negative, true_negative + false_positive)
        per_class[label] = {
            "support": support,
            "predicted": sum(confusion.get(other, {}).get(label, 0) for other in matrix_labels if other in confusion),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "balanced_accuracy": _mean([recall, specificity]),
            "wilson_recall_95": wilson_interval(true_positive, support),
        }

    supports = {label: details["support"] for label, details in per_class.items()}
    class_metrics = list(per_class.values())
    macro = {
        "precision": _mean(item["precision"] for item in class_metrics),
        "recall": _mean(item["recall"] for item in class_metrics),
        "f1": _mean(item["f1"] for item in class_metrics),
        "balanced_accuracy": _mean(item["recall"] for item in class_metrics),
    }
    weighted = {
        name: _rate(
            sum((item[name] or 0.0) * item["support"] for item in class_metrics),
            total,
        )
        for name in ("precision", "recall", "f1")
    }
    micro_precision = _rate(correct, total)
    micro = {
        "precision": micro_precision,
        "recall": micro_precision,
        "f1": micro_precision,
    }
    majority = max(supports.values(), default=0)
    return {
        "count": total,
        "correct": correct,
        "accuracy": _rate(correct, total),
        "parse_rate": _rate(
            sum(label != UNPARSED_LABEL for label in predicted_labels), total
        ),
        "labels": target_labels,
        "matrix_labels": matrix_labels,
        "support": supports,
        "majority_baseline": _rate(majority, total),
        "imbalance_ratio": (
            max(supports.values()) / min(supports.values())
            if supports and min(supports.values())
            else None
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "macro": macro,
        "weighted": weighted,
        "micro": micro,
    }


def _canonical(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def list_metrics(
    records: Iterable[dict[str, Any]],
    expected_field: str,
    predicted_field: str,
) -> dict[str, Any]:
    """Calculate set precision/recall for list-valued answers."""

    rows = list(records)
    examples: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected_answer", {})
        prediction = row.get("parsed_prediction") or {}
        expected_items = {_canonical(item) for item in expected.get(expected_field, [])}
        predicted_items = {
            _canonical(item) for item in prediction.get(predicted_field, [])
        }
        true_positive = len(expected_items & predicted_items)
        false_positive = len(predicted_items - expected_items)
        false_negative = len(expected_items - predicted_items)
        precision = _rate(true_positive, true_positive + false_positive) or 0.0
        recall = _rate(true_positive, true_positive + false_negative) or 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        examples.append(
            {
                "id": row.get("id"),
                "expected_count": len(expected_items),
                "predicted_count": len(predicted_items),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "set_exact": expected_items == predicted_items,
                "empty_case": not expected_items,
            }
        )

    empty = [item for item in examples if item["empty_case"]]
    nonempty = [item for item in examples if not item["empty_case"]]
    true_positive = sum(item["true_positive"] for item in nonempty)
    false_positive = sum(item["false_positive"] for item in nonempty)
    false_negative = sum(item["false_negative"] for item in nonempty)
    micro_precision = _rate(true_positive, true_positive + false_positive)
    micro_recall = _rate(true_positive, true_positive + false_negative)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision is not None
        and micro_recall is not None
        and micro_precision + micro_recall
        else None
    )
    empty_exact = _rate(sum(item["set_exact"] for item in empty), len(empty))
    nonempty_exact = _rate(sum(item["set_exact"] for item in nonempty), len(nonempty))
    return {
        "count": len(examples),
        "empty_cases": len(empty),
        "nonempty_cases": len(nonempty),
        "empty_exact_rate": empty_exact,
        "nonempty_exact_rate": nonempty_exact,
        "balanced_exact_rate": _mean([empty_exact, nonempty_exact]),
        "macro": {
            "precision": _mean(item["precision"] for item in nonempty),
            "recall": _mean(item["recall"] for item in nonempty),
            "f1": _mean(item["f1"] for item in nonempty),
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "totals": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "examples": examples,
    }


def boolean_metrics(
    records: Iterable[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    """Summarize a boolean outcome with positive/negative balance."""

    rows = list(records)
    positives = sum(row.get(field) is True for row in rows)
    negatives = sum(row.get(field) is False for row in rows)
    return {
        "count": len(rows),
        "positive_rate": _rate(positives, len(rows)),
        "positive": positives,
        "negative": negatives,
        "support": {"true": positives, "false": negatives},
        "majority_baseline": _rate(max(positives, negatives), len(rows)),
    }


def enrich_deduction_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add normalized target/prediction fields used by the generic UI."""

    enriched = []
    for record in records:
        item = dict(record)
        item["target_class"] = record.get("target_rule")
        item["predicted_class"] = record.get("predicted_rule")
        enriched.append(item)
    return enriched


def enrich_representation_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for record in records:
        item = dict(record)
        prediction = record.get("parsed_prediction") or {}
        item["target_class"] = record.get("task")
        item["predicted_class"] = prediction.get("task")
        enriched.append(item)
    return enriched


def scalar_summary(records: Iterable[dict[str, Any]], fields: Iterable[str]) -> dict[str, float | None]:
    rows = list(records)
    return {
        field: _rate(sum(row.get(field) is True for row in rows), len(rows))
        for field in fields
    }


def group_boolean_rates(
    records: Iterable[dict[str, Any]], group_key: str, fields: Iterable[str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(group_key, "unknown"))].append(record)
    result = {}
    for group, rows in sorted(groups.items()):
        result[group] = {
            "count": len(rows),
            **scalar_summary(rows, fields),
            "mean_generation_seconds": _mean(
                [row.get("generation_seconds") for row in rows]
            ),
        }
    return result
