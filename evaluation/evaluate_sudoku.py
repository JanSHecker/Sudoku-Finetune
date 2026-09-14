"""Evaluate the pinned 4-bit Qwen3.5-2B model on Sudoku.

This evaluator is intentionally independent of the BANKING77 adapter pipeline:
it loads the base checkpoint only, uses a frozen zero-shot prompt, and writes
raw per-example predictions so formatting failures remain inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "LiquidAI/LFM2.5-2.6B": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
}
MAX_INPUT_TOKENS = 512
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
# Nine rows of nine digit tokens plus eight newline tokens. Keeping the cap at
# the first complete grid prevents a model that does not emit EOS from
# appending another grid and being rejected as malformed.
DEFAULT_MAX_NEW_TOKENS = 89


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/sudoku-eval-v1/eval-v1.metadata.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/sudoku-eval-v1/predictions.base-4bit.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/sudoku-eval-v1/metrics.base-4bit.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional saved PEFT adapter to attach to the quantized base model.",
    )
    parser.add_argument(
        "--prompt",
        choices=("direct", "rules"),
        default="direct",
        help="Prompt protocol to evaluate.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            puzzle = row.get("puzzle")
            solution = row.get("solution")
            if (
                not isinstance(item_id, str)
                or item_id in seen
                or not isinstance(puzzle, str)
                or len(puzzle) != 81
                or any(character not in "0123456789" for character in puzzle)
                or not isinstance(solution, str)
                or len(solution) != 81
                or any(character not in "123456789" for character in solution)
            ):
                raise ValueError(f"invalid or duplicate dataset row at {path}:{line_number}")
            seen.add(item_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['id']}|{row.get('band')}|{row['puzzle']}|{row['solution']}|{row.get('score')}\n".encode()
        )
    return digest.hexdigest()


DIRECT_SYSTEM_PROMPT = (
    "Solve the standard 9x9 Sudoku puzzle. Reply with exactly 81 digits in "
    "row-major order. Use only digits 1-9. Do not include zeroes, spaces, "
    "punctuation, explanations, Markdown, or any other text."
)


RULES_SYSTEM_PROMPT = (
    "Solve a standard 9x9 Sudoku. Sudoku rules: the grid has nine rows and "
    "nine columns, divided into nine 3x3 boxes. Every row must contain each "
    "digit 1 through 9 exactly once. Every column must contain each digit 1 "
    "through 9 exactly once. Every 3x3 box must contain each digit 1 through "
    "9 exactly once. Digits already given in the puzzle are fixed and must "
    "not be changed. Fill every empty cell so that all three constraints hold. "
    "Reply with exactly 81 solution digits in row-major order. Use only digits "
    "1-9. Do not include zeroes, spaces, punctuation, explanations, Markdown, "
    "or any other text."
)

PROMPTS = {
    "direct": ("sudoku-zero-shot-v1", DIRECT_SYSTEM_PROMPT),
    "rules": ("sudoku-rules-zero-shot-v1", RULES_SYSTEM_PROMPT),
}


def prompt_messages(puzzle: str, system_prompt: str) -> list[dict[str, str]]:
    rows = "\n".join(puzzle[index : index + 9] for index in range(0, 81, 9))
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Solve this puzzle. 0 denotes an empty cell:\n" + rows,
        },
    ]


def is_lfm_model(model_id: str) -> bool:
    return model_id.casefold().startswith("liquidai/lfm2")


def chat_template_kwargs(model_id: str) -> dict[str, bool]:
    if is_lfm_model(model_id):
        return {}
    return {"enable_thinking": False}


def model_revision(model_id: str, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    return MODEL_REVISIONS.get(model_id)


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(
    model_id: str, revision: str | None, local_files_only: bool
) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 4-bit baseline")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The 4-bit baseline requires BF16 GPU support")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config(),
        device_map={"": 0},
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def generate_batch(
    model: Any,
    tokenizer: Any,
    puzzles: list[str],
    system_prompt: str,
    max_new_tokens: int,
    model_id: str,
) -> list[dict[str, Any]]:
    prompts = [
        tokenizer.apply_chat_template(
            prompt_messages(puzzle, system_prompt),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(model_id),
        )
        for puzzle in puzzles
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise ValueError("Sudoku prompt exceeded the 512-token input budget")
    device = next(model.parameters()).device
    inputs = inputs.to(device)
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
    results: list[dict[str, Any]] = []
    for tokens in generated:
        token_ids = tokens.detach().cpu().tolist()
        raw_output = tokenizer.decode(token_ids, skip_special_tokens=False)
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if is_lfm_model(model_id) and "</think>" in generated_text:
            generated_text = generated_text.rsplit("</think>", 1)[-1].strip()
        results.append(
            {
                "generated_token_ids": token_ids,
                "raw_output": raw_output,
                "generated_text": generated_text,
            }
        )
    return results


def valid_solution(value: str) -> bool:
    if len(value) != 81 or any(character not in "123456789" for character in value):
        return False
    for row in range(9):
        if len(set(value[row * 9 : row * 9 + 9])) != 9:
            return False
    for column in range(9):
        if len({value[row * 9 + column] for row in range(9)}) != 9:
            return False
    for box_row in range(3):
        for box_column in range(3):
            cells = [
                value[(box_row * 3 + row) * 9 + box_column * 3 + column]
                for row in range(3)
                for column in range(3)
            ]
            if len(set(cells)) != 9:
                return False
    return True


def score_prediction(puzzle: str, solution: str, generated_text: str) -> dict[str, Any]:
    stripped = generated_text.strip()
    normalized = "".join(stripped.split())
    parseable = len(normalized) == 81 and all(character in "123456789" for character in normalized)
    valid = parseable and valid_solution(normalized)
    clues_preserved = parseable and all(
        clue == "0" or clue == normalized[index]
        for index, clue in enumerate(puzzle)
    )
    exact = normalized == solution
    strict_exact = stripped == solution
    blank_total = puzzle.count("0") if parseable else 0
    blank_correct = (
        sum(puzzle[index] == "0" and normalized[index] == solution[index] for index in range(81))
        if parseable
        else 0
    )
    return {
        "normalized_output": normalized,
        "parseable": parseable,
        "valid_solution": valid,
        "clues_preserved": clues_preserved,
        "valid_with_clues": valid and clues_preserved,
        "exact_solution": exact,
        "strict_exact_solution": strict_exact,
        "blank_cells": blank_total,
        "correct_blank_cells": blank_correct,
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        parseable = sum(item["parseable"] for item in group)
        valid = sum(item["valid_solution"] for item in group)
        clues = sum(item["clues_preserved"] for item in group)
        valid_with_clues = sum(item["valid_with_clues"] for item in group)
        exact = sum(item["exact_solution"] for item in group)
        strict = sum(item["strict_exact_solution"] for item in group)
        blank_total = sum(item["blank_cells"] for item in group)
        blank_correct = sum(item["correct_blank_cells"] for item in group)

        def fraction(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        return {
            "count": count,
            "parse_rate": fraction(parseable, count),
            "valid_solution_rate": fraction(valid, count),
            "clue_preservation_rate_among_parseable": fraction(clues, parseable),
            "valid_with_clues_rate": fraction(valid_with_clues, count),
            "exact_solution_accuracy": fraction(exact, count),
            "strict_exact_accuracy": fraction(strict, count),
            "exact_accuracy_ci95": wilson_interval(exact, count),
            "blank_cell_accuracy_among_parseable": fraction(blank_correct, blank_total),
        }

    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_band[record["band"]].append(record)
    return {"overall": one(records), "by_band": {band: one(rows) for band, rows in sorted(by_band.items())}}


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
                raise ValueError(f"prediction file has a different dataset at line {line_number}")
            existing[record["id"]] = record
    return existing


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--batch-size and --max-new-tokens must be greater than zero")
    rows = read_dataset(args.dataset, args.limit)
    dataset_hash = dataset_fingerprint(rows)
    prompt_version, system_prompt = PROMPTS[args.prompt]
    if args.predictions.exists() and not args.resume and not args.overwrite:
        raise SystemExit("prediction file exists; use --resume or --overwrite")
    if args.overwrite and args.predictions.exists():
        args.predictions.unlink()
    existing = load_existing(args.predictions, dataset_hash) if args.resume else {}
    pending = [row for row in rows if row["id"] not in existing]

    started = time.perf_counter()
    model = None
    if pending:
        revision = model_revision(args.model_id, args.revision)
        model, tokenizer = load_base_model(args.model_id, revision, args.local_files_only)
        if args.adapter is not None:
            if not args.adapter.exists():
                raise FileNotFoundError(f"adapter directory does not exist: {args.adapter}")
            model = PeftModel.from_pretrained(model, str(args.adapter))
        torch.cuda.reset_peak_memory_stats()
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                generation_started = time.perf_counter()
                generations = generate_batch(
                    model,
                    tokenizer,
                    [row["puzzle"] for row in batch],
                    system_prompt,
                    args.max_new_tokens,
                    args.model_id,
                )
                generation_seconds = time.perf_counter() - generation_started
                for row, generation in zip(batch, generations):
                    scored = score_prediction(
                        row["puzzle"], row["solution"], generation["generated_text"]
                    )
                    record = {
                        "id": row["id"],
                        "band": row.get("band"),
                        "puzzle": row["puzzle"],
                        "solution": row["solution"],
                        "score": row.get("score"),
                        "dataset_sha256": dataset_hash,
                        "model_id": args.model_id,
                        "model_revision": revision,
                        "quantization": "4-bit NF4, double quantization, BF16 compute",
                        "prompt_variant": args.prompt,
                        "prompt_version": prompt_version,
                        "generation": {
                            "do_sample": False,
                            "repetition_penalty": 1.0,
                            "max_new_tokens": args.max_new_tokens,
                        },
                        "generation_seconds_batch": generation_seconds,
                        **generation,
                        **scored,
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    existing[row["id"]] = record
                stream.flush()
                print(f"evaluated {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)

    ordered_records = [existing[row["id"]] for row in rows]
    elapsed = time.perf_counter() - started
    metrics = {
        "schema": "sudoku-baseline-metrics-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "model_id": args.model_id,
        "model_revision": model_revision(args.model_id, args.revision),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "quantization": "4-bit NF4, double quantization, BF16 compute",
        "prompt_variant": args.prompt,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "input_format": "nine rows of nine digits; 0 denotes an empty cell",
        "decoding": {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "examples": len(ordered_records),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
        ),
        **summarize(ordered_records),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
