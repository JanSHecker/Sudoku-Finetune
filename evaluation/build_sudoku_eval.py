"""Build a frozen, stratified Sudoku evaluation set from generator metadata.

The input files are JSONL files emitted by the Rust generator with ``--metadata``.
This script deliberately keeps the answer key in the artifact directory while
the evaluator sends only the puzzle field to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BANDS = (
    ("b1", 1_000, 1_999),
    ("b2", 2_000, 3_999),
    ("b3", 4_000, 7_999),
    ("b4", 8_000, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Generator metadata JSONL file; repeat for multiple candidate files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/validation/sudoku-eval-v1"),
    )
    parser.add_argument(
        "--generator-binary",
        type=Path,
        help="Optional generator executable to hash into the manifest.",
    )
    parser.add_argument("--per-band", type=int, default=250)
    return parser.parse_args()


def read_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
                puzzle = item.get("puzzle")
                solution = item.get("solution")
                score = item.get("score")
                if (
                    not isinstance(puzzle, str)
                    or len(puzzle) != 81
                    or any(character not in "0123456789" for character in puzzle)
                    or not isinstance(solution, str)
                    or len(solution) != 81
                    or any(character not in "123456789" for character in solution)
                    or not isinstance(score, int)
                ):
                    raise ValueError(f"invalid candidate record in {path}:{line_number}")
                candidates.setdefault(puzzle, item)
    return list(candidates.values())


def band_for_score(score: int) -> tuple[str, int, int | None] | None:
    for band, lower, upper in BANDS:
        if score >= lower and (upper is None or score <= upper):
            return band, lower, upper
    return None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.per_band <= 0:
        raise SystemExit("--per-band must be greater than zero")

    candidates = read_candidates(args.input)
    by_band: dict[str, list[dict[str, Any]]] = {band: [] for band, _, _ in BANDS}
    for candidate in candidates:
        band = band_for_score(candidate["score"])
        if band is not None:
            by_band[band[0]].append(candidate)

    selected_by_band: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for band, _, _ in BANDS:
        # Sorting makes selection independent of input file order and parallel
        # generation scheduling. Exact puzzle deduplication happened on ingest.
        band_candidates = sorted(
            by_band[band], key=lambda item: (item["score"], item["puzzle"])
        )
        if len(band_candidates) < args.per_band:
            missing.append(f"{band}: {len(band_candidates)}/{args.per_band}")
            continue
        selected_by_band[band] = [
            {**item, "band": band} for item in band_candidates[: args.per_band]
        ]
    if missing:
        raise SystemExit(
            "not enough candidates for score bands: " + ", ".join(missing)
        )

    # Interleave bands so a small smoke run exercises the full difficulty range
    # instead of silently evaluating only the easiest prefix.
    selected = [
        selected_by_band[band][index]
        for index in range(args.per_band)
        for band, _, _ in BANDS
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_lines: list[str] = []
    metadata_lines: list[str] = []
    band_indices: dict[str, int] = {band: 0 for band, _, _ in BANDS}
    for item in selected:
        band = item["band"]
        item_id = f"sudoku-v1-{band}-{band_indices[band]:04d}"
        band_indices[band] += 1
        tsv_lines.append(f"{item['puzzle']}\t{item['solution']}\n")
        metadata_lines.append(
            json.dumps(
                {
                    "schema": "sudoku-eval-v1",
                    "id": item_id,
                    "band": band,
                    "puzzle": item["puzzle"],
                    "solution": item["solution"],
                    "score": item["score"],
                    "clues": item.get("clues"),
                    "nodes": item.get("nodes"),
                    "backtracks": item.get("backtracks"),
                    "guesses": item.get("guesses"),
                    "max_depth": item.get("max_depth"),
                    "source_attempt": item.get("attempt"),
                    "source_seed": item.get("seed"),
                },
                sort_keys=True,
            )
            + "\n"
        )

    tsv_bytes = "".join(tsv_lines).encode()
    metadata_bytes = "".join(metadata_lines).encode()
    input_hash = hashlib.sha256()
    for path in sorted(args.input):
        input_hash.update(path.read_bytes())
    manifest = {
        "schema": "sudoku-eval-manifest-v1",
        "evaluation_set": "sudoku-eval-v1",
        "count": len(selected),
        "per_band": args.per_band,
        "bands": {band: args.per_band for band, _, _ in BANDS},
        "input_files": [str(path) for path in sorted(args.input)],
        "input_sha256": input_hash.hexdigest(),
        "tsv_sha256": sha256_bytes(tsv_bytes),
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "generator_binary": str(args.generator_binary) if args.generator_binary else None,
        "generator_binary_sha256": (
            sha256_file(args.generator_binary) if args.generator_binary else None
        ),
        "format": "headerless TSV: puzzle\\tsolution",
        "puzzle_rows": "nine rows of nine digits; 0 denotes an empty cell",
        "selection": "exact-string deduplication, then score ascending within each band",
    }

    (args.output_dir / "eval-v1.tsv").write_bytes(tsv_bytes)
    (args.output_dir / "eval-v1.metadata.jsonl").write_bytes(metadata_bytes)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
