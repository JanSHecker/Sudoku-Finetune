"""Train a QLoRA model to produce sound, structured Sudoku deductions.

The input is the state-level dataset emitted by
``evaluation/build_sudoku_deduction_dataset.py``.  Its puzzle-level split is
preserved: only rows marked ``train`` are used for training and only rows
marked ``validation`` are used for validation.  The separate benchmark file
is never read by this trainer.

Examples::

    python train_sudoku.py inspect
    python train_sudoku.py smoke --model-id Qwen/Qwen3.5-2B
    python train_sudoku.py smoke --model-id LiquidAI/LFM2.5-2.6B
    python train_sudoku.py run --model-id Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "LiquidAI/LFM2.5-2.6B": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
}
DEFAULT_INPUT = Path("artifacts/sudoku-deductions-v1.jsonl")
SEED = 42
MAX_LENGTH = 1024
SYSTEM_PROMPT = (
    "You are a Sudoku deduction engine. Identify one logically sound deduction "
    "from the supplied state. Do not guess or provide a final solution. Return "
    "only the requested JSON deduction."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "smoke", "run"))
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Deduction dataset JSONL; repeat to combine datasets.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_row(row: dict[str, Any], source: Path, line_number: int) -> None:
    target = row.get("target_deduction")
    completion = row.get("completion")
    if (
        row.get("schema") != "sudoku-deduction-example"
        or row.get("schema_version") != 1
        or not isinstance(row.get("id"), str)
        or not isinstance(row.get("puzzle"), str)
        or len(row["puzzle"]) != 81
        or any(character not in "0123456789" for character in row["puzzle"])
        or row.get("split") not in {"train", "validation", "test"}
        or not isinstance(row.get("prompt"), str)
        or not row["prompt"]
        or not isinstance(target, dict)
        or not isinstance(completion, str)
    ):
        raise ValueError(f"invalid deduction example at {source}:{line_number}")
    try:
        if json.loads(completion) != target:
            raise ValueError("completion does not match target_deduction")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid completion at {source}:{line_number}") from error


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_row(row, path, line_number)
                item_id = row["id"]
                if item_id in rows and rows[item_id] != row:
                    raise ValueError(f"conflicting duplicate id: {item_id}")
                rows[item_id] = row
    if not rows:
        raise ValueError("no deduction examples were loaded")
    return list(rows.values())


def split_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_split = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    puzzle_sets = {name: {row["puzzle"] for row in group} for name, group in by_split.items()}
    if puzzle_sets["train"] & puzzle_sets["validation"]:
        raise ValueError("train and validation contain the same puzzle")
    if puzzle_sets["train"] & puzzle_sets["test"]:
        raise ValueError("train and test contain the same puzzle")
    if puzzle_sets["validation"] & puzzle_sets["test"]:
        raise ValueError("validation and test contain the same puzzle")
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("dataset must contain non-empty train and validation splits")
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


def is_lfm_model(model_id: str) -> bool:
    return model_id.casefold().startswith("liquidai/lfm2")


def chat_template_kwargs(model_id: str) -> dict[str, bool]:
    if is_lfm_model(model_id):
        return {}
    return {"enable_thinking": False}


def format_dataset(dataset: Dataset, tokenizer: Any, model_id: str) -> Dataset:
    """Render the evaluator-compatible conversation with completion-only loss."""

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        kwargs = chat_template_kwargs(model_id)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        if is_lfm_model(model_id):
            # LFM opens a mandatory thinking block in its generation template.
            return {"prompt": prompt, "completion": "</think>\n" + example["completion"]}

        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": example["completion"]}],
            tokenize=False,
            **kwargs,
        )
        if not full.startswith(prompt):
            raise ValueError("chat template did not preserve the prompt prefix")
        return {"prompt": prompt, "completion": full[len(prompt) :]}

    return dataset.map(format_example, remove_columns=dataset.column_names)


def training_dataset(rows: list[dict[str, Any]], tokenizer: Any, model_id: str) -> Dataset:
    records = [
        {"prompt": row["prompt"], "completion": row["completion"]}
        for row in rows
    ]
    return format_dataset(Dataset.from_list(records), tokenizer, model_id)


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_id: str, revision: str | None) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for QLoRA training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this QLoRA configuration requires a BF16-capable GPU")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config(),
        device_map={"": 0},
    )
    return model, tokenizer


def model_revision(model_id: str, requested: str | None) -> str | None:
    return requested if requested is not None else MODEL_REVISIONS.get(model_id)


def default_output_dir(model_id: str) -> Path:
    names = {
        "Qwen/Qwen3.5-2B": "qwen3.5-2b",
        "LiquidAI/LFM2.5-2.6B": "lfm2.5-2.6b",
    }
    name = names.get(model_id, model_id.rsplit("/", 1)[-1].casefold())
    return Path("artifacts") / f"sudoku-deduction-{name}-qlora"


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["id"]):
        digest.update(
            f"{row['id']}|{row['puzzle']}|{row['target_rule']}|{row['completion']}\n".encode()
        )
    return digest.hexdigest()


def inspect(args: argparse.Namespace) -> None:
    inputs = args.input or [DEFAULT_INPUT]
    rows = read_jsonl(inputs)
    train_rows, validation_rows, counts = split_rows(rows)
    print(
        json.dumps(
            {
                "model": args.model_id,
                "model_revision": model_revision(args.model_id, args.revision),
                "inputs": [str(path) for path in inputs],
                "counts": counts,
                "train_rules": sorted({row["target_rule"] for row in train_rows}),
                "validation_rules": sorted({row["target_rule"] for row in validation_rows}),
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

    smoke = args.command == "smoke"
    if smoke:
        train_rows = train_rows[: min(64, len(train_rows))]
        validation_rows = validation_rows[: min(32, len(validation_rows))]
    output_root = args.output_dir or default_output_dir(args.model_id)
    output_dir = output_root.with_name(
        output_root.name + "-smoke" if smoke else output_root.name
    )
    revision = model_revision(args.model_id, args.revision)

    started = time.perf_counter()
    model, tokenizer = load_model(args.model_id, revision)
    torch.cuda.reset_peak_memory_stats()
    train_dataset = training_dataset(train_rows, tokenizer, args.model_id)
    validation_dataset = training_dataset(validation_rows, tokenizer, args.model_id)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=1 if smoke else args.epochs,
            max_steps=10 if smoke else -1,
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
            save_strategy="no",
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
    result = trainer.train()
    validation = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "schema": "sudoku-deduction-qlora-training-v1",
        "model": args.model_id,
        "model_revision": revision,
        "input_files": [str(path) for path in inputs],
        "dataset_fingerprint": fingerprint(train_rows + validation_rows),
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
            "max_steps": 10 if smoke else -1,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "completion_only_loss": True,
        },
        "parameters": {"trainable": trainable, "total": total},
        "training": result.metrics,
        "validation": validation,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": metrics["counts"], "validation": validation}, indent=2))
    print(f"Saved adapter and metrics to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        inspect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
