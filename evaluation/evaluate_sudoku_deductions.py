"""Evaluate a language model on structured Sudoku deductions.

The evaluator accepts any deduction listed as valid for the state and also
independently checks the predicted witness and candidate-state transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from validate_sudoku_deduction_dataset import (
    apply_deduction,
    validate_witness,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "LiquidAI/LFM2.5-2.6B": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
}
SYSTEM_PROMPT = (
    "You are a Sudoku deduction engine. Identify one logically sound deduction "
    "from the supplied state. Do not guess or provide a final solution. Return "
    "only the requested JSON deduction."
)
MAX_INPUT_TOKENS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/sudoku-deduction-benchmark-v1.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/sudoku-deduction-benchmark-v1.predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/sudoku-deduction-benchmark-v1.metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--provider", choices=("local", "openrouter"), default="local")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Use this UTF-8 system prompt instead of the built-in v1 prompt.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file used by the OpenRouter provider.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_system_prompt(path: Path | None) -> str:
    if path is None:
        return SYSTEM_PROMPT
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"system prompt file is empty: {path}")
    return prompt


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid environment entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"empty environment key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def openrouter_config() -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from the environment")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is missing from the environment")
    return {
        "api_key": api_key,
        "model": model,
        "base_url": os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        "site_url": os.environ.get("OPENROUTER_SITE_URL", "").strip(),
        "app_name": os.environ.get("OPENROUTER_APP_NAME", "Sudoku deduction benchmark"),
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "512")),
        "reasoning_effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "").strip(),
        "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0")),
        "timeout": float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "180")),
        "max_retries": int(os.environ.get("OPENROUTER_MAX_RETRIES", "5")),
        "request_delay": float(
            os.environ.get("OPENROUTER_REQUEST_DELAY_SECONDS", "0.25")
        ),
    }


def read_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            if (
                not isinstance(item_id, str)
                or item_id in seen
                or row.get("schema") != "sudoku-deduction-example"
                or row.get("schema_version") != 1
                or not isinstance(row.get("prompt"), str)
                or not isinstance(row.get("target_deduction"), dict)
                or not isinstance(row.get("valid_deductions"), list)
            ):
                raise ValueError(f"invalid benchmark row at {path}:{line_number}")
            seen.add(item_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


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


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['id']}|{row['puzzle']}|{row['target_rule']}\n".encode())
    return digest.hexdigest()


def load_model(
    model_id: str,
    revision: str | None,
    local_files_only: bool,
    adapter: Path | None,
) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for the 4-bit evaluator")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the evaluator requires a BF16-capable GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        local_files_only=local_files_only,
    )
    if adapter is not None:
        if not adapter.exists():
            raise FileNotFoundError(f"adapter directory does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def model_prompts(rows: list[dict[str, Any]]) -> list[str]:
    return [
        row["prompt"]
        for row in rows
    ]


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    system_prompt: str,
    model_id: str,
) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(model_id),
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise ValueError("deduction prompt exceeded the input token budget")
    inputs = inputs.to(next(model.parameters()).device)
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


def generate_openrouter(
    prompt: str, system_prompt: str, config: dict[str, Any]
) -> str:
    request_body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": config["max_tokens"],
        "stream": False,
        "temperature": config["temperature"],
        "response_format": {"type": "json_object"},
    }
    if config["reasoning_effort"]:
        request_body["reasoning"] = {
            "effort": config["reasoning_effort"],
            "exclude": True,
        }
    payload = json.dumps(request_body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    if config["site_url"]:
        headers["HTTP-Referer"] = config["site_url"]
    if config["app_name"]:
        headers["X-Title"] = config["app_name"]
    url = f"{config['base_url']}/chat/completions"
    for attempt in range(config["max_retries"] + 1):
        try:
            request = Request(url, data=payload, headers=headers, method="POST")
            with urlopen(request, timeout=config["timeout"]) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"OpenRouter response has no choices: {body}")
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            if not isinstance(content, str) or not content.strip():
                finish_reason = choices[0].get("finish_reason")
                if attempt >= config["max_retries"]:
                    raise RuntimeError(
                        "OpenRouter response did not contain text content "
                        f"after retries (finish_reason={finish_reason!r})"
                    )
                time.sleep(min(2**attempt, 60.0))
                continue
            return content
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt >= config["max_retries"]:
                raise RuntimeError(
                    f"OpenRouter HTTP {error.code}: {detail[:1000]}"
                ) from error
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(min(delay, 60.0))
        except (URLError, TimeoutError) as error:
            if attempt >= config["max_retries"]:
                raise RuntimeError(f"OpenRouter request failed: {error}") from error
            time.sleep(min(2**attempt, 60.0))
    raise RuntimeError("OpenRouter request exhausted retries")


def parse_prediction(text: str, model_id: str = MODEL_ID) -> dict[str, Any] | None:
    if is_lfm_model(model_id) and "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def rule_key(deduction: dict[str, Any]) -> str | None:
    technique = deduction.get("technique")
    if not isinstance(technique, dict) or not isinstance(technique.get("kind"), str):
        return None
    if technique["kind"] != "fish":
        return technique["kind"]
    if not isinstance(technique.get("size"), int) or not isinstance(
        technique.get("orientation"), str
    ):
        return None
    return f"fish_{technique['size']}_{technique['orientation']}"


def score_prediction(
    row: dict[str, Any], raw_output: str, model_id: str = MODEL_ID
) -> dict[str, Any]:
    prediction = parse_prediction(raw_output, model_id)
    result: dict[str, Any] = {
        "raw_output": raw_output,
        "parseable": prediction is not None,
        "sound_deduction": False,
        "valid_deduction": False,
        "valid_rule": False,
        "exact_target": False,
        "predicted_rule": rule_key(prediction) if prediction else None,
    }
    if prediction is None:
        return result
    state = row["state"]
    board = state["board"]
    values = [
        token[1:-1] if token.startswith("[") and token.endswith("]") else token
        for line in state["candidate_grid"]
        for token in line.split()
    ]
    try:
        validate_witness(board, values, prediction)
        apply_deduction(board, values, prediction)
    except (KeyError, TypeError, ValueError):
        return result
    result["sound_deduction"] = True
    valid = row["valid_deductions"]
    result["valid_deduction"] = prediction in valid
    result["valid_rule"] = result["predicted_rule"] in {
        rule_key(item) for item in valid
    }
    result["exact_target"] = prediction == row["target_deduction"]
    result["parsed_prediction"] = prediction
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def fraction(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "count": count,
            "parse_rate": fraction(sum(item["parseable"] for item in group), count),
            "sound_deduction_rate": fraction(
                sum(item["sound_deduction"] for item in group), count
            ),
            "valid_deduction_rate": fraction(
                sum(item["valid_deduction"] for item in group), count
            ),
            "valid_rule_rate": fraction(sum(item["valid_rule"] for item in group), count),
            "exact_target_rate": fraction(sum(item["exact_target"] for item in group), count),
            "mean_generation_seconds": (
                sum(item["generation_seconds"] for item in group) / count if count else 0.0
            ),
        }

    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rule[record["target_rule"]].append(record)
    return {
        "overall": one(records),
        "by_rule": {rule: one(rows) for rule, rows in sorted(by_rule.items())},
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
                raise ValueError(f"prediction file has a different dataset at line {line_number}")
            existing[record["id"]] = record
    return existing


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--batch-size and --max-new-tokens must be greater than zero")
    load_env(args.env_file)
    openrouter = args.provider == "openrouter"
    config = openrouter_config() if openrouter else None
    if openrouter and args.batch_size != 1:
        raise SystemExit("--batch-size must be 1 when --provider openrouter is used")
    rows = read_dataset(args.dataset, args.limit)
    system_prompt = read_system_prompt(args.system_prompt_file)
    dataset_hash = dataset_fingerprint(rows)
    if args.predictions.exists() and not args.resume and not args.overwrite:
        raise SystemExit("prediction file exists; use --resume or --overwrite")
    if args.overwrite and args.predictions.exists():
        args.predictions.unlink()
    existing = load_existing(args.predictions, dataset_hash) if args.resume else {}
    pending = [row for row in rows if row["id"] not in existing]

    model_id = config["model"] if config is not None else args.model_id
    revision = None if openrouter else model_revision(args.model_id, args.revision)
    started = time.perf_counter()
    model = None
    if pending:
        if openrouter:
            tokenizer = None
        else:
            model, tokenizer = load_model(
                args.model_id, revision, args.local_files_only, args.adapter
            )
            torch.cuda.reset_peak_memory_stats()
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                generation_started = time.perf_counter()
                if openrouter:
                    outputs = [
                        generate_openrouter(batch[0]["prompt"], system_prompt, config)
                    ]
                else:
                    outputs = generate_batch(
                        model,
                        tokenizer,
                        model_prompts(batch),
                        args.max_new_tokens,
                        system_prompt,
                        args.model_id,
                    )
                generation_seconds = time.perf_counter() - generation_started
                for row, raw_output in zip(batch, outputs):
                    scored = score_prediction(row, raw_output, model_id)
                    record = {
                        "id": row["id"],
                        "dataset_sha256": dataset_hash,
                        "target_rule": row["target_rule"],
                        "provider": args.provider,
                        "model_id": model_id,
                        "model_revision": revision,
                        "adapter": str(args.adapter) if args.adapter is not None else None,
                        "system_prompt": system_prompt,
                        "system_prompt_file": (
                            str(args.system_prompt_file)
                            if args.system_prompt_file is not None
                            else None
                        ),
                        "generation": {
                            "do_sample": False,
                            "repetition_penalty": 1.0,
                            "max_new_tokens": (
                                config["max_tokens"] if openrouter else args.max_new_tokens
                            ),
                            "reasoning_effort": (
                                config["reasoning_effort"] if openrouter else None
                            ),
                        },
                        "generation_seconds": generation_seconds,
                        **scored,
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    existing[row["id"]] = record
                stream.flush()
                print(f"evaluated {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)
                if openrouter and start + args.batch_size < len(pending):
                    time.sleep(config["request_delay"])

    ordered = [existing[row["id"]] for row in rows]
    metrics = {
        "schema": "sudoku-deduction-metrics-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "provider": args.provider,
        "model_id": model_id,
        "model_revision": revision,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "system_prompt": system_prompt,
        "system_prompt_file": (
            str(args.system_prompt_file) if args.system_prompt_file is not None else None
        ),
        "examples": len(ordered),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if model is not None and torch.cuda.is_available()
            else None
        ),
        **summarize(ordered),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
