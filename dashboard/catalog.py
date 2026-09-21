"""Artifact discovery and normalization for the local dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    classification_metrics,
    enrich_deduction_records,
    enrich_representation_records,
    group_boolean_rates,
    list_metrics,
    scalar_summary,
)


@dataclass(frozen=True)
class RunArtifact:
    """A metrics file and the optional prediction file associated with it."""

    path: Path
    data: dict[str, Any]
    kind: str
    predictions_path: Path | None

    @property
    def schema(self) -> str:
        return str(self.data.get("schema", "unknown"))

    @property
    def model(self) -> str:
        return str(self.data.get("model_id") or self.data.get("model") or "unknown")

    @property
    def adapter(self) -> str | None:
        value = self.data.get("adapter")
        return Path(str(value)).name if value else None

    @property
    def provider(self) -> str:
        return str(self.data.get("provider") or "local")

    @property
    def split(self) -> str | None:
        value = self.data.get("split")
        return str(value) if value else None

    @property
    def display_name(self) -> str:
        model = self.model.split("/")[-1]
        if self.path.name == "metrics.json":
            suffix = self.path.parent.name
        else:
            suffix = self.path.name[: -len(".metrics.json")] if self.path.name.endswith(".metrics.json") else self.kind
        provider = f" · {self.provider}" if self.provider != "local" else ""
        return f"{model} · {suffix}{provider}"

    @property
    def run_id(self) -> str:
        return str(self.path)


def _kind(data: dict[str, Any]) -> str:
    schema = str(data.get("schema", ""))
    if "deduction" in schema and "metrics" in schema:
        return "deduction"
    if "representation-generation" in schema:
        return "representation"
    if "baseline" in schema:
        return "baseline"
    if "qlora-training" in schema or "run-v1" in schema:
        return "training"
    return "other"


def _candidate_prediction_paths(metrics_path: Path, data: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    explicit = data.get("predictions") or data.get("predictions_path")
    if explicit:
        candidates.append(Path(str(explicit)))
    name = metrics_path.name
    if name.endswith(".metrics.json"):
        candidates.append(metrics_path.with_name(name[: -len(".metrics.json")] + ".predictions.jsonl"))
    if name == "metrics.json":
        candidates.append(metrics_path.with_name("predictions.jsonl"))
    if name.startswith("metrics.") and name.endswith(".json"):
        candidates.append(metrics_path.with_name("predictions" + name[len("metrics") : -len(".json")] + ".jsonl"))
    return candidates


def _resolve_path(path: Path, workspace_root: Path) -> Path:
    if path.is_absolute():
        return path
    direct = workspace_root / path
    return direct if direct.exists() else path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover_runs(artifact_root: str | Path = "artifacts") -> list[RunArtifact]:
    root = Path(artifact_root)
    if not root.exists():
        return []
    workspace_root = root.parent
    paths = sorted(
        set(root.rglob("*.metrics.json"))
        | set(root.rglob("metrics.json"))
        | set(root.rglob("metrics.*.json")),
        key=str,
    )
    runs: list[RunArtifact] = []
    for path in paths:
        data = _read_json(path)
        if data is None:
            continue
        predictions_path = next(
            (
                resolved
                for candidate in _candidate_prediction_paths(path, data)
                for resolved in [_resolve_path(candidate, workspace_root)]
                if resolved.exists()
            ),
            None,
        )
        runs.append(RunArtifact(path, data, _kind(data), predictions_path))
    return runs


def load_predictions(run: RunArtifact) -> list[dict[str, Any]]:
    if run.predictions_path is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with run.predictions_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _deduction_summary(run: RunArtifact, records: list[dict[str, Any]]) -> dict[str, Any]:
    records = enrich_deduction_records(records)
    classification = classification_metrics(records, "target_class", "predicted_class")
    fields = (
        "parseable",
        "sound_deduction",
        "valid_deduction",
        "valid_rule",
        "exact_target",
    )
    return {
        "classification": classification,
        "rates": scalar_summary(records, fields),
        "by_class": group_boolean_rates(records, "target_class", fields),
        "mean_generation_seconds": (
            sum(float(row.get("generation_seconds", 0.0)) for row in records) / len(records)
            if records
            else None
        ),
        "prediction_records": records,
    }


def _representation_summary(run: RunArtifact, records: list[dict[str, Any]]) -> dict[str, Any]:
    records = enrich_representation_records(records)
    classification = classification_metrics(records, "target_class", "predicted_class")
    fields = ("parseable", "schema_valid", "task_correct", "exact_answer")
    by_task = group_boolean_rates(records, "target_class", fields)
    list_summaries = {}
    for task, expected_field, predicted_field in (
        ("candidate_locations", "cells", "cells"),
        ("naked_single_scan", "singles", "singles"),
    ):
        task_records = [row for row in records if row.get("task") == task]
        if task_records:
            list_summaries[task] = list_metrics(task_records, expected_field, predicted_field)
    return {
        "classification": classification,
        "rates": scalar_summary(records, fields),
        "by_class": by_task,
        "lists": list_summaries,
        "mean_generation_seconds": (
            sum(float(row.get("generation_seconds", 0.0)) for row in records) / len(records)
            if records
            else None
        ),
        "prediction_records": records,
    }


def summarize_run(run: RunArtifact) -> dict[str, Any]:
    """Return a common summary, deriving richer metrics when predictions exist."""

    records = load_predictions(run)
    if run.kind == "deduction" and records:
        derived = _deduction_summary(run, records)
    elif run.kind == "representation" and records:
        derived = _representation_summary(run, records)
    else:
        derived = {
            "classification": None,
            "rates": {},
            "by_class": {},
            "lists": {},
            "prediction_records": records,
        }
    return {
        "run": run,
        "derived": derived,
        "overall": run.data.get("overall", {}),
        "training": run.data.get("training", {}),
        "validation": run.data.get("validation", {}),
        "data": run.data,
    }


def artifact_fingerprint(artifact_root: str | Path = "artifacts") -> tuple[tuple[str, int, int], ...]:
    """Provide a cache key that changes when metrics or prediction files change."""

    root = Path(artifact_root)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.endswith(".metrics.json")
            or path.name.startswith("metrics.")
            or path.name.endswith(".predictions.jsonl")
        )
    ]
    return tuple(sorted((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files))
