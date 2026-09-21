"""Train a QLoRA adapter on Sudoku representation tasks.

This is the isolated Phase 3 warm-up process.  It learns to read coordinates,
units, peers, and candidate sets before a later deduction-specific trainer
continues from the saved adapter.

Examples::

    python train_sudoku_representation.py inspect
    python train_sudoku_representation.py smoke --model-id Qwen/Qwen3.5-2B
    python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from train_sudoku import (
    chat_template_kwargs,
    is_lfm_model,
    load_model,
    model_revision,
)


DEFAULT_INPUT = Path("artifacts/datasets/training/representation/sudoku-representation-v1.jsonl")
DEFAULT_OUTPUT = Path("artifacts/checkpoints/representation/sudoku-representation-qwen3.5-2b-qlora")
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SAVE_STEPS = 10
DEFAULT_SAVE_TOTAL_LIMIT = 3
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
CHECKPOINT_STATE_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "smoke", "run"))
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Limit optimizer steps; use -1 for the configured epoch count.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Start a new optimizer run from a compatible saved LoRA adapter.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="Save a resumable checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
        help="Keep at most this many recent checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint in the output directory.",
    )
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                completion = row.get("completion")
                if (
                    row.get("schema") != "sudoku-representation-example"
                or row.get("schema_version") not in {1, 2}
                    or not isinstance(row.get("id"), str)
                    or row.get("split") not in {"train", "validation", "test"}
                    or not isinstance(row.get("source_puzzle"), str)
                    or len(row["source_puzzle"]) != 81
                    or not isinstance(row.get("task"), str)
                    or not isinstance(row.get("system_prompt"), str)
                    or not isinstance(row.get("prompt"), str)
                    or not isinstance(completion, str)
                ):
                    raise ValueError(f"invalid representation row at {path}:{line_number}")
                try:
                    answer = json.loads(completion)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid completion at {path}:{line_number}") from error
                if not isinstance(answer, dict) or answer.get("task") != row["task"]:
                    raise ValueError(f"completion task mismatch at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise ValueError("no representation examples were loaded")
    return rows


def split_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_split = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    puzzle_sets = {name: {row["source_puzzle"] for row in group} for name, group in by_split.items()}
    if puzzle_sets["train"] & puzzle_sets["validation"]:
        raise ValueError("train and validation contain the same puzzle")
    if puzzle_sets["train"] & puzzle_sets["test"]:
        raise ValueError("train and test contain the same puzzle")
    if puzzle_sets["validation"] & puzzle_sets["test"]:
        raise ValueError("validation and test contain the same puzzle")
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("representation dataset needs train and validation rows")
    counts = {
        "rows": len(rows),
        "train": len(by_split["train"]),
        "validation": len(by_split["validation"]),
        "test": len(by_split["test"]),
        "train_puzzles": len(puzzle_sets["train"]),
        "validation_puzzles": len(puzzle_sets["validation"]),
        "test_puzzles": len(puzzle_sets["test"]),
    }
    return by_split["train"], by_split["validation"], counts


def format_dataset(
    rows: list[dict[str, Any]], tokenizer: Any, model_id: str
) -> Dataset:
    formatted: list[dict[str, str]] = []
    for row in rows:
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt"]},
        ]
        kwargs = chat_template_kwargs(model_id)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if is_lfm_model(model_id):
            completion = "</think>\n" + row["completion"]
        else:
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": row["completion"]}],
                tokenize=False,
                **kwargs,
            )
            if not full.startswith(prompt):
                raise ValueError("chat template did not preserve the prompt prefix")
            completion = full[len(prompt) :]
        formatted.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(formatted)


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["id"]):
        digest.update(
            f"{row.get('schema_version', 1)}|{row['id']}|{row['source_puzzle']}|"
            f"{row['task']}|{row['completion']}|{row.get('system_prompt', '')}\n".encode()
        )
    return digest.hexdigest()


def checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    found = [
        path
        for path in output_dir.iterdir()
        if path.is_dir()
        and CHECKPOINT_PATTERN.fullmatch(path.name)
        and all((path / filename).is_file() for filename in CHECKPOINT_STATE_FILES)
        and (any(path.glob("*.safetensors")) or any(path.glob("*.bin")))
    ]
    return sorted(
        found,
        key=lambda path: int(CHECKPOINT_PATTERN.fullmatch(path.name).group(1)),
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    found = checkpoints(output_dir)
    return found[-1] if found else None


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_configuration(
    args: argparse.Namespace,
    inputs: list[Path],
    revision: str | None,
    dataset_fingerprint: str,
    counts: dict[str, int],
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema": "sudoku-representation-run-v1",
        "model": args.model_id,
        "model_revision": revision,
        "input_files": [str(path) for path in inputs],
        "dataset_fingerprint": dataset_fingerprint,
        "counts": counts,
        "smoke": smoke,
        "epochs": 1 if smoke else args.epochs,
        "max_steps": 10 if smoke else args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "seed": args.seed,
        "initial_adapter": str(args.initial_adapter)
        if args.initial_adapter is not None
        else None,
    }


def validate_resume_configuration(
    path: Path, expected: dict[str, Any]
) -> None:
    if not path.exists():
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read resume metadata from {path}") from error
    stored_comparable = {key: value for key, value in stored.items() if key != "max_steps"}
    expected_comparable = {
        key: value for key, value in expected.items() if key != "max_steps"
    }
    if stored_comparable != expected_comparable:
        raise ValueError(
            "resume configuration does not match the existing run; "
            "use the same model, dataset, and training settings"
        )


def write_run_status(
    output_dir: Path,
    status: str,
    started_at: str,
    **details: Any,
) -> None:
    write_json_atomically(
        output_dir / "run_status.json",
        {
            "schema": "sudoku-representation-run-status-v1",
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def inspect(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input or [DEFAULT_INPUT])
    train_rows, validation_rows, counts = split_rows(rows)
    print(
        json.dumps(
            {
                "model": args.model_id,
                "model_revision": model_revision(args.model_id, args.revision),
                "inputs": [str(path) for path in (args.input or [DEFAULT_INPUT])],
                "counts": counts,
                "train_tasks": sorted({row["task"] for row in train_rows}),
                "validation_tasks": sorted({row["task"] for row in validation_rows}),
                "example": {
                    "prompt": train_rows[0]["prompt"],
                    "completion": train_rows[0]["completion"],
                },
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    inputs = args.input or [DEFAULT_INPUT]
    rows = read_jsonl(inputs)
    train_rows, validation_rows, counts = split_rows(rows)
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch and gradient accumulation must be positive")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.max_length <= 0:
        raise ValueError("epochs, learning rate, and max length must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max steps must be -1 or positive")
    if args.save_steps <= 0 or args.save_total_limit <= 0:
        raise ValueError("save steps and save total limit must be positive")
    if args.initial_adapter is not None:
        if not args.initial_adapter.exists():
            raise FileNotFoundError(
                f"initial adapter directory does not exist: {args.initial_adapter}"
            )
        adapter_config = args.initial_adapter / "adapter_config.json"
        if not adapter_config.is_file():
            raise ValueError(f"initial adapter has no adapter_config.json: {args.initial_adapter}")
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
            raise ValueError("initial adapter must be a causal-language-model LoRA adapter")
        if config.get("r") != 16 or config.get("lora_alpha") != 32:
            raise ValueError("initial adapter does not match the representation LoRA configuration")
        base_model = config.get("base_model_name_or_path")
        if base_model and base_model != args.model_id:
            raise ValueError(
                f"initial adapter was trained for {base_model}, not {args.model_id}"
            )

    smoke = args.command == "smoke"
    if smoke:
        train_rows = train_rows[: min(64, len(train_rows))]
        validation_rows = validation_rows[: min(32, len(validation_rows))]
    output_root = args.output_dir or DEFAULT_OUTPUT
    output_dir = output_root.with_name(
        output_root.name + "-smoke" if smoke else output_root.name
    )
    revision = model_revision(args.model_id, args.revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_fingerprint = fingerprint(train_rows + validation_rows)
    configuration = run_configuration(
        args, inputs, revision, dataset_fingerprint, counts, smoke
    )
    configuration_path = output_dir / "run_config.json"
    resume_checkpoint = latest_checkpoint(output_dir)
    if args.resume:
        if resume_checkpoint is None:
            raise FileNotFoundError(
                f"no checkpoint found in {output_dir}; cannot resume"
            )
        validate_resume_configuration(configuration_path, configuration)
    elif resume_checkpoint is not None:
        raise FileExistsError(
            f"checkpoints already exist in {output_dir}; rerun with --resume"
        )
    elif configuration_path.exists() and (output_dir / "metrics.json").exists():
        raise FileExistsError(
            f"a completed run already exists in {output_dir}; "
            "choose a new output directory"
        )
    else:
        write_json_atomically(configuration_path, configuration)

    started_at = datetime.now(timezone.utc).isoformat()
    write_run_status(
        output_dir,
        "running",
        started_at,
        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
    )

    started = time.perf_counter()
    model, tokenizer = load_model(args.model_id, revision)
    if args.initial_adapter is not None:
        model = PeftModel.from_pretrained(
            model, str(args.initial_adapter), is_trainable=True
        )
    train_dataset = format_dataset(train_rows, tokenizer, args.model_id)
    validation_dataset = format_dataset(validation_rows, tokenizer, args.model_id)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=(
            None
            if args.initial_adapter is not None
            else LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=1 if smoke else args.epochs,
            max_steps=10 if smoke else args.max_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_length=args.max_length,
            completion_only_loss=True,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no",
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_only_model=False,
            ignore_data_skip=False,
            logging_steps=1 if smoke else 10,
            report_to="none",
            seed=args.seed,
        ),
    )
    trainable = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    try:
        result = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint else None
            )
        )
        validation = trainer.evaluate()

        trainer.model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        metrics = {
            "schema": "sudoku-representation-qlora-training-v1",
            "model": args.model_id,
            "model_revision": revision,
            "input_files": [str(path) for path in inputs],
            "dataset_fingerprint": dataset_fingerprint,
            "counts": counts
            | {
                "train_used": len(train_rows),
                "validation_used": len(validation_rows),
                "smoke": smoke,
            },
            "configuration": {
                "quantization": "4-bit NF4 with double quantization and BF16 compute",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
                "epochs": 1 if smoke else args.epochs,
                "max_steps": 10 if smoke else args.max_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_length": args.max_length,
                "completion_only_loss": True,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
                "initial_adapter": str(args.initial_adapter)
                if args.initial_adapter is not None
                else None,
            },
            "parameters": {"trainable": trainable, "total": total},
            "training": result.metrics,
            "validation": validation,
            "runtime_seconds": time.perf_counter() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        write_json_atomically(output_dir / "metrics.json", metrics)
        final_checkpoint = latest_checkpoint(output_dir)
        write_run_status(
            output_dir,
            "completed",
            started_at,
            checkpoint=str(final_checkpoint) if final_checkpoint else None,
        )
        print(json.dumps({"counts": metrics["counts"], "validation": validation}, indent=2))
        print(f"Saved representation adapter and metrics to {output_dir}")
    except KeyboardInterrupt:
        checkpoint = latest_checkpoint(output_dir)
        write_run_status(
            output_dir,
            "interrupted",
            started_at,
            checkpoint=str(checkpoint) if checkpoint else None,
        )
        print(f"Training interrupted. Resume with --resume from {output_dir}")
        raise
    except Exception as error:
        checkpoint = latest_checkpoint(output_dir)
        write_run_status(
            output_dir,
            "failed",
            started_at,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            checkpoint=str(checkpoint) if checkpoint else None,
        )
        print(
            f"Training failed; existing checkpoints were preserved. "
            f"Resume with --resume from {output_dir}"
        )
        raise


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        inspect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
