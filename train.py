"""Train and evaluate a QLoRA intent classifier on BANKING77.

The model is still used as a causal language model: it receives a chat prompt
containing a banking-support message and generates one of the 77 intent labels.
QLoRA keeps the pretrained base weights frozen and 4-bit quantized while small
LoRA adapter matrices are trained in higher precision.

The command is deliberately positional so inspecting the data cannot
accidentally start training::

    python train.py inspect
    python train.py evaluate
    python train.py smoke
    python train.py run

``inspect`` only prepares and prints dataset metadata. ``evaluate`` loads an
already-saved adapter. ``smoke`` and ``run`` are the only commands that call
the trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "mteb/banking77"
SEED = 42
MAX_LENGTH = 512
OUTPUT_DIR = Path("artifacts/qwen2.5-1.5b-banking77-qlora")


def load_splits() -> tuple[DatasetDict, list[str]]:
    """Load BANKING77 and create the reproducible train/validation/test splits.

    BANKING77 already provides a training split and an official test split. We
    keep that official test split completely untouched so it remains a fair
    final evaluation set. A stratified sample of 1,000 rows is instead removed
    from the original training split for validation. Stratification preserves
    approximately the same frequency of every intent in both new splits.

    The human-readable ``label_text`` values are treated as the canonical
    output vocabulary. Sorting them makes the order deterministic, which is
    useful because that exact order is embedded in every system prompt and in
    the metrics file.

    Returns:
        A pair containing:

        - A ``DatasetDict`` with ``train``, ``validation``, and untouched
          ``test`` splits.
        - The sorted list of all 77 allowed intent-label strings.

    Raises:
        ValueError: If the dataset no longer contains exactly 77 training
            labels, or if the test split contains a label unseen in training.

    Loading may read from the local Hugging Face cache or download the dataset
    when it is not cached; it never loads model weights or uses the GPU.
    """
    raw = load_dataset(DATASET_ID)
    labels = sorted(set(raw["train"]["label_text"]))
    if len(labels) != 77 or set(raw["test"]["label_text"]) - set(labels):
        raise ValueError("BANKING77 must contain the same 77 labels in train and test")

    # ``train_test_split`` can stratify only on a ClassLabel column. The
    # dataset's numeric ``label`` values are therefore encoded explicitly
    # before asking Hugging Face Datasets for the seeded split.
    train = raw["train"].class_encode_column("label")
    split = train.train_test_split(
        test_size=1000,
        seed=SEED,
        stratify_by_column="label",
    )
    return DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    ), labels


def build_system_prompt(labels: list[str]) -> str:
    """Build the instruction shared by training and inference.

    Every allowed label is written directly into the prompt. This turns an
    open-ended generation problem into a constrained routing task and teaches
    the model the exact spelling expected by the evaluator. The instruction
    also forbids explanations because accuracy uses exact string equality: an
    otherwise-correct response such as ``"The intent is card_arrival"`` would
    be counted as invalid.

    Args:
        labels: Canonical intent names. Their order should be deterministic so
            prompt tokenization is identical across runs.

    Returns:
        One system-message string containing the task and valid label list.
    """
    return (
        "Classify the online-banking support request. Reply with exactly one "
        "intent label and nothing else. Valid labels: "
        + ", ".join(labels)
    )


def prompt_messages(text: str, system_prompt: str) -> list[dict[str, str]]:
    """Represent one unlabelled request as the input side of a chat.

    The assistant response is intentionally absent. During training it is
    appended separately as the completion; during evaluation the same two
    messages are followed by an empty assistant-generation marker produced by
    the tokenizer's chat template. Keeping one helper for both paths prevents
    train/inference prompt drift.

    Args:
        text: The customer's banking-support message.
        system_prompt: The classification instruction from
            :func:`build_system_prompt`.

    Returns:
        A two-message chat containing the system instruction and user request.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def format_dataset(dataset: Dataset, system_prompt: str, tokenizer) -> Dataset:
    """Convert raw BANKING77 rows into prompt/completion training records.

    ``SFTTrainer`` receives separate ``prompt`` and ``completion`` strings.
    With ``completion_only_loss=True`` this separation is important: tokens in
    the long system/user prompt provide context but do not contribute to the
    loss, while the short assistant label is the only target the optimizer is
    asked to predict.

    Both strings are produced with the model tokenizer's own chat template so
    special tokens exactly match Qwen's expected format. We serialize the
    prompt once with an assistant-generation marker, then serialize the full
    conversation with the gold assistant label. The completion is the suffix
    that remains after removing the prompt prefix; it therefore includes only
    the assistant-side content and any closing chat tokens.

    Args:
        dataset: A split with at least ``text`` and ``label_text`` columns.
        system_prompt: Shared instruction listing the valid labels.
        tokenizer: Qwen tokenizer exposing ``apply_chat_template``.

    Returns:
        A new dataset containing only ``prompt`` and ``completion`` columns.

    Raises:
        ValueError: If the tokenizer renders the labelled conversation in a
            way that does not begin with the inference prompt. Slicing would be
            unsafe in that case and could train on the wrong tokens.
    """
    def format_example(example: dict) -> dict:
        """Format one raw row; called by ``Dataset.map`` for every example."""
        messages = prompt_messages(example["text"], system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_and_completion = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": example["label_text"]}],
            tokenize=False,
        )
        # The suffix operation below is correct only when the two independent
        # template renders share an identical prompt prefix. Fail loudly if a
        # future tokenizer/chat-template version changes that assumption.
        if not prompt_and_completion.startswith(prompt):
            raise ValueError("The chat template did not preserve the prompt prefix")
        return {
            "prompt": prompt,
            "completion": prompt_and_completion[len(prompt) :],
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


def inspect_dataset() -> None:
    """Print a safe, human-readable preview of dataset preparation.

    This command deliberately does not instantiate a tokenizer or model. It
    displays split sizes, the complete label vocabulary, and one conceptual
    chat example as JSON so the data contract can be reviewed before any GPU
    work. The shown prompt/completion are message objects rather than Qwen's
    final tokenized chat string; :func:`format_dataset` performs that later
    once a tokenizer has been loaded.

    The function returns nothing and writes its report to standard output.
    """
    splits, labels = load_splits()
    system_prompt = build_system_prompt(labels)
    example = splits["train"][0]
    formatted = {
        "prompt": prompt_messages(example["text"], system_prompt),
        "completion": [{"role": "assistant", "content": example["label_text"]}],
    }
    print(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "splits": {name: len(split) for name, split in splits.items()},
                "label_count": len(labels),
                "labels": labels,
                "formatted_example": formatted,
            },
            indent=2,
        )
    )


def quantization_config() -> BitsAndBytesConfig:
    """Describe how bitsandbytes should load the frozen base model.

    This is QLoRA's ``Q``: pretrained linear-layer weights are stored in the
    4-bit NormalFloat (NF4) representation to reduce VRAM use. NF4 is designed
    for the roughly normal distribution commonly found in neural-network
    weights. Double quantization also quantizes the scale constants used by
    the first quantization step, saving a little more memory.

    Arithmetic is not performed in 4-bit. Quantized values are dequantized as
    needed and matrix operations use BF16 compute. The LoRA adapter parameters
    created later are trainable higher-precision tensors; this configuration
    does not turn the final model or its activations into a universally 4-bit
    system.

    Returns:
        A Transformers ``BitsAndBytesConfig`` passed directly to
        ``AutoModelForCausalLM.from_pretrained``.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model():
    """Load the tokenizer and 4-bit base model onto the first CUDA GPU.

    The explicit hardware checks produce understandable errors before the much
    larger model-loading operation begins. BF16 support is required because it
    is the configured computation dtype for quantized layers and training.

    Qwen does not necessarily define a distinct padding token. Batched
    generation needs one, so the end-of-sequence token is reused for padding;
    its attention-mask positions remain ignored and it introduces no new
    vocabulary entry.

    ``device_map={"": 0}`` places the complete model on CUDA device 0. The
    returned model contains 4-bit frozen base weights. LoRA adapters are not
    attached until ``SFTTrainer`` receives a ``LoraConfig`` or until
    ``PeftModel.from_pretrained`` loads a saved adapter.

    Returns:
        ``(model, tokenizer)`` for the configured Qwen checkpoint.

    Raises:
        RuntimeError: If CUDA is unavailable or the selected GPU cannot perform
            BF16 computation.

    The first call may download model/tokenizer files into the Hugging Face
    cache; later calls normally reuse that cache.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for smoke and run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a BF16-capable GPU")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config(),
        device_map={"": 0},
    )
    return model, tokenizer


def generate_labels(
    model,
    tokenizer,
    texts: list[str],
    system_prompt: str,
    batch_size: int,
) -> list[str]:
    """Generate one raw intent-label string for each input message.

    Evaluation deliberately uses deterministic greedy decoding
    (``do_sample=False``). This makes repeated evaluations comparable and is
    appropriate because the task has one expected label rather than many
    creative answers. Predictions are returned exactly as generated except for
    surrounding whitespace; they are not coerced to the nearest valid label.
    That allows ``valid_label_rate`` to expose formatting failures honestly.

    Decoder-only models need left padding for mixed-length batched generation:
    every row must end at its actual prompt before new tokens are appended.
    ``use_cache`` is enabled to reuse attention keys/values and speed up
    autoregressive decoding. Both settings, plus the model's train/eval state,
    are restored in ``finally`` so calling evaluation during a training
    workflow cannot silently change subsequent behavior.

    Args:
        model: Base or PEFT-wrapped causal language model already on CUDA.
        tokenizer: Matching tokenizer with a configured padding token.
        texts: Customer messages to classify, in desired output order.
        system_prompt: Classification instruction and label vocabulary.
        batch_size: Number of prompts generated together. Larger values can
            improve throughput but require more VRAM.

    Returns:
        Generated strings in the same order as ``texts``.

    Notes:
        Inputs are truncated to ``MAX_LENGTH`` tokens. Generation is capped at
        24 new tokens, which is ample for the longest BANKING77 label while
        limiting accidental verbose output.
    """
    predictions: list[str] = []

    # Query the model instead of assuming ``cuda:0`` so this helper also works
    # with a PEFT wrapper whose exposed device is inherited from its base model.
    device = next(model.parameters()).device

    # Evaluation temporarily changes mutable tokenizer/model state. Save every
    # value first so the function is safe to call before, during, or after
    # training.
    old_padding_side = tokenizer.padding_side
    old_use_cache = model.config.use_cache
    was_training = model.training

    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()
    try:
        # ``inference_mode`` disables gradient tracking and version counters.
        # Autocast makes non-quantized operations consistently use BF16 on GPU.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(texts), batch_size):
                # Render exactly the same system/user structure used when the
                # training prompt was created, ending at Qwen's assistant turn.
                prompts = [
                    tokenizer.apply_chat_template(
                        prompt_messages(text, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for text in texts[start : start + batch_size]
                ]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to(device)
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=24,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # ``generate`` returns prompt tokens followed by new tokens.
                # All batch rows have the same padded prompt width, so slicing
                # at the input tensor width removes the prompt from every row.
                generated = output[:, inputs["input_ids"].shape[1] :]
                predictions.extend(
                    text.strip()
                    for text in tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )
                )
    finally:
        # Restoration also runs if tokenization or generation raises, avoiding
        # a hard-to-find state leak into training or another evaluation call.
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_use_cache
        if was_training:
            model.train()

    return predictions


def classification_metrics(
    gold: list[str], predicted: list[str], labels: list[str]
) -> dict:
    """Calculate exact-label aggregate and per-intent classification metrics.

    A prediction is correct only when its complete stripped string exactly
    equals the gold label. Matching is case-sensitive and punctuation-sensitive.
    Outputs outside the supplied label vocabulary remain in the prediction
    counts and confusion list, but can never be true positives. This mirrors
    deployment: a router cannot use a label it does not recognize.

    Per-label precision asks, "Of everything predicted as this intent, how much
    was correct?" Recall asks, "Of every example belonging to this intent, how
    much did the model find?" F1 is their harmonic mean. A zero denominator is
    defined as 0.0 rather than producing an exception or NaN.

    The macro metrics are unweighted arithmetic means over all 77 labels, so a
    rare intent contributes just as much as a frequent one. Accuracy instead
    counts correct examples globally. BANKING77's test set is close to balanced,
    but keeping both views makes the evaluator reusable and exposes uneven
    intent performance.

    Args:
        gold: Expected label string for each example.
        predicted: Generated label string for each corresponding example.
        labels: Complete canonical vocabulary used for validity checks and the
            per-label report.

    Returns:
        A JSON-serializable dictionary containing example count, accuracy,
        valid-label rate, macro precision/recall/F1, detailed per-label scores,
        and the ten most frequent ``expected -> predicted`` mistakes.

    Raises:
        ValueError: If the lists are empty or have different lengths.
    """
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and predicted labels must have the same non-zero length")

    # Counter-based one-vs-rest accounting avoids an extra metrics dependency
    # and makes the exact treatment of invalid generated labels explicit.
    label_set = set(labels)
    support = Counter(gold)
    predicted_count = Counter(predicted)
    correct = Counter(
        expected for expected, actual in zip(gold, predicted) if expected == actual
    )
    per_label = {}
    for label in labels:
        # ``correct[label]`` is the true-positive count for this class.
        # ``predicted_count`` supplies TP + FP; ``support`` supplies TP + FN.
        precision = correct[label] / predicted_count[label] if predicted_count[label] else 0.0
        recall = correct[label] / support[label] if support[label] else 0.0
        per_label[label] = {
            "support": support[label],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }

    # Keep the raw generated value as ``actual`` even when it is not one of the
    # 77 labels; invalid output patterns are useful diagnostic information.
    confusions = Counter(
        (expected, actual)
        for expected, actual in zip(gold, predicted)
        if expected != actual
    )
    return {
        "examples": len(gold),
        "accuracy": sum(a == b for a, b in zip(gold, predicted)) / len(gold),
        "valid_label_rate": sum(item in label_set for item in predicted) / len(gold),
        "macro_precision": sum(item["precision"] for item in per_label.values()) / len(labels),
        "macro_recall": sum(item["recall"] for item in per_label.values()) / len(labels),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "per_label": per_label,
        "top_confusions": [
            {"expected": expected, "predicted": actual, "count": count}
            for (expected, actual), count in confusions.most_common(10)
        ],
    }


def evaluate_classifier(
    model,
    tokenizer,
    dataset: Dataset,
    labels: list[str],
    system_prompt: str,
    batch_size: int,
) -> dict:
    """Run end-to-end generation and scoring for one dataset split.

    This small composition is the common evaluation path for the untouched
    base model, the in-memory trained adapter, and an adapter reloaded from
    disk. Reusing it ensures every comparison uses identical prompting,
    decoding, and metric rules.

    Args:
        model: Causal language model to evaluate.
        tokenizer: Tokenizer matching the base checkpoint.
        dataset: Split containing ``text`` and ``label_text`` columns.
        labels: Complete valid-label vocabulary.
        system_prompt: Shared routing instruction.
        batch_size: Generation batch size.

    Returns:
        The metrics dictionary produced by :func:`classification_metrics`.
    """
    predicted = generate_labels(
        model,
        tokenizer,
        list(dataset["text"]),
        system_prompt,
        batch_size,
    )
    return classification_metrics(list(dataset["label_text"]), predicted, labels)


def evaluate_saved_models(batch_size: int) -> None:
    """Re-evaluate the base model and a previously saved adapter on all tests.

    The existing ``metrics.json`` acts as a safety check that a completed
    training artifact exists at ``OUTPUT_DIR``. The base checkpoint is loaded
    and measured first. ``PeftModel.from_pretrained`` then attaches the saved
    LoRA weights to that same base model, avoiding a second copy of the 1.5B
    checkpoint in GPU memory, and the full untouched test split is measured
    again.

    Detailed results replace the ``baseline`` and ``fine_tuned`` sections in
    the existing metrics file. Training metadata already present in the file is
    preserved. This command never updates weights or invokes ``trainer.train``;
    it only performs generation, metric calculation, and a metrics-file write.

    Args:
        batch_size: Number of test prompts decoded simultaneously. Increase it
            for throughput only when GPU memory allows.

    Raises:
        FileNotFoundError: If the expected full-run ``metrics.json`` is absent.
    """
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing trained adapter metrics: {metrics_path}")

    splits, labels = load_splits()
    system_prompt = build_system_prompt(labels)
    model, tokenizer = load_base_model()
    started = time.perf_counter()
    baseline = evaluate_classifier(
        model, tokenizer, splits["test"], labels, system_prompt, batch_size
    )
    # PEFT loads only the small saved adapter and connects it to the already
    # resident quantized base model; it does not merge or retrain the weights.
    model = PeftModel.from_pretrained(model, str(OUTPUT_DIR))
    fine_tuned = evaluate_classifier(
        model, tokenizer, splits["test"], labels, system_prompt, batch_size
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["baseline"] = baseline
    metrics["fine_tuned"] = fine_tuned
    metrics["detailed_evaluation_runtime_seconds"] = time.perf_counter() - started
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    def summary(result: dict) -> dict:
        """Reduce a full report to console-friendly aggregates and weak labels."""
        worst = sorted(
            result["per_label"].items(), key=lambda item: item[1]["recall"]
        )[:10]
        return {
            key: result[key]
            for key in ("accuracy", "valid_label_rate", "macro_precision", "macro_recall", "macro_f1")
        } | {"lowest_recall": [{"label": label, **scores} for label, scores in worst]}

    print(json.dumps({"baseline": summary(baseline), "fine_tuned": summary(fine_tuned)}, indent=2))
    print(f"Updated {metrics_path}")


def smoke_splits(splits: DatasetDict) -> DatasetDict:
    """Create small deterministic subsets for a short pipeline smoke test.

    Each source split is independently shuffled with the project seed before
    selecting rows. Shuffling avoids taking an ordered prefix that might contain
    only a narrow group of intents. The smoke test is intentionally too small
    to establish final quality; its job is to verify that loading, formatting,
    quantized training, evaluation, and adapter saving all work end to end.

    Args:
        splits: Full train/validation/test ``DatasetDict`` from
            :func:`load_splits`.

    Returns:
        A new ``DatasetDict`` with 256 training, 64 validation, and 100 test
        examples. The original datasets are not modified.
    """
    return DatasetDict(
        train=splits["train"].shuffle(seed=SEED).select(range(256)),
        validation=splits["validation"].shuffle(seed=SEED).select(range(64)),
        test=splits["test"].shuffle(seed=SEED).select(range(100)),
    )


def train(command: str, batch_size: int, gradient_accumulation: int) -> None:
    """Run the smoke or full QLoRA workflow and save its adapter and metrics.

    The workflow deliberately evaluates the base model before modifying it, so
    the baseline and fine-tuned scores come from the same checkpoint, prompt,
    test rows, batch size, and decoding code. It then converts only the training
    and validation splits to prompt/completion records, attaches LoRA adapters
    through ``SFTTrainer``, trains, evaluates validation loss, and finally runs
    exact-label generation on the untouched test split.

    ``smoke`` changes three things: it uses the small subsets from
    :func:`smoke_splits`, stops after ten optimizer steps, and saves under a
    separate ``-smoke`` directory. ``run`` uses every training row for two
    epochs. This separation prevents a diagnostic run from overwriting the
    full adapter.

    Args:
        command: Either ``"smoke"`` or ``"run"``. The CLI choices enforce
            this before the function is called.
        batch_size: Per-device number of examples for training, validation,
            and generation. This is the largest direct activation-memory knob.
        gradient_accumulation: Number of micro-batches whose gradients are
            accumulated before one optimizer update. With the defaults, the
            effective training batch is ``4 * 4 = 16`` examples on one GPU.

    Side effects:
        Loads model weights onto CUDA, performs generation and training, and
        writes adapter files, tokenizer files, and ``metrics.json`` below the
        selected artifact directory.
    """
    smoke = command == "smoke"
    splits, labels = load_splits()
    if smoke:
        splits = smoke_splits(splits)

    system_prompt = build_system_prompt(labels)
    # Smoke artifacts have their own location so a later full evaluation never
    # mistakes a ten-step diagnostic adapter for the completed training run.
    output_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "-smoke") if smoke else OUTPUT_DIR
    started = time.perf_counter()
    model, tokenizer = load_base_model()

    # Reset after model loading so reported peak VRAM describes this workflow's
    # active CUDA high-water mark from this point, not earlier work in process.
    torch.cuda.reset_peak_memory_stats()

    # Measure before SFTTrainer attaches LoRA parameters. This is the frozen,
    # quantized Qwen checkpoint's zero-shot performance on the same test rows.
    baseline = evaluate_classifier(
        model,
        tokenizer,
        splits["test"],
        labels,
        system_prompt,
        batch_size,
    )

    # Only train/validation require serialized completions. Test examples stay
    # raw because evaluation must prompt the model without revealing labels.
    train_dataset = format_dataset(splits["train"], system_prompt, tokenizer)
    validation_dataset = format_dataset(splits["validation"], system_prompt, tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=LoraConfig(
            # Rank controls adapter capacity and memory. Rank 16 adds two small
            # trainable low-rank matrices around each targeted linear layer.
            r=16,
            # Alpha scales the adapter update by alpha/rank. Here 32/16 = 2.
            lora_alpha=32,
            # Adapter dropout regularizes the learned update, not base weights.
            lora_dropout=0.05,
            # Apply LoRA broadly to transformer linear layers rather than naming
            # only attention projections; this improves capacity at modest cost
            # for a model of this size.
            target_modules="all-linear",
            # Do not train an additional full-size bias vector for each layer.
            bias="none",
            # The underlying task remains next-token causal language modelling.
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            # Trainer bookkeeping points at the final adapter directory, but
            # automatic checkpoints are disabled below; only the explicit final
            # save near the end of this function is retained.
            output_dir=str(output_dir),
            # The smoke path favors speed; the real run sees the full data twice.
            num_train_epochs=1 if smoke else 2,
            # A positive max_steps overrides epochs, so -1 leaves the two-epoch
            # full run alone while smoke exits after exactly ten updates.
            max_steps=10 if smoke else -1,
            # Common LoRA learning rates are higher than full-model fine-tuning
            # rates because only newly initialized adapter weights are updated.
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # Accumulation increases effective batch size without keeping every
            # example's activations in memory simultaneously.
            gradient_accumulation_steps=gradient_accumulation,
            # Prompts longer than this are truncated; 512 covers the short
            # BANKING77 messages plus the comparatively long 77-label prompt.
            max_length=MAX_LENGTH,
            # Mask system and user tokens from the loss. The optimizer learns
            # only to generate the assistant completion containing the label.
            completion_only_loss=True,
            # BF16 has FP32-like exponent range and is stable on supported GPUs.
            bf16=True,
            # Recompute selected forward activations during backward instead of
            # storing them all, exchanging extra compute for much lower VRAM.
            gradient_checkpointing=True,
            # The non-reentrant implementation is the recommended modern path
            # and behaves better with PEFT/frozen parameters.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # Skip evaluation during epochs to save time. Validation is still
            # run once explicitly after training below.
            eval_strategy="no",
            # Intermediate checkpoints are unnecessary for this learning run;
            # the final adapter is explicitly saved after all evaluations.
            save_strategy="no",
            logging_steps=1 if smoke else 25,
            # Keep the project local and avoid implicit Weights & Biases or
            # similar integrations changing external state.
            report_to="none",
            seed=SEED,
        ),
    )

    # PEFT marks only LoRA tensors trainable. Recording both counts makes the
    # small trainable fraction visible in the resulting experiment metadata.
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())

    # This is the only call that updates parameters. Validation and final test
    # evaluation below run after the adapter has learned.
    train_result = trainer.train()
    validation = trainer.evaluate()
    fine_tuned = evaluate_classifier(
        trainer.model,
        tokenizer,
        splits["test"],
        labels,
        system_prompt,
        batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Because ``trainer.model`` is PEFT-wrapped, ``save_pretrained`` stores the
    # LoRA adapter/config rather than duplicating the full Qwen base checkpoint.
    # The tokenizer is saved beside it so inference uses the identical chat
    # template and special-token settings.
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Keep configuration, quality, time, parameter counts, and measured memory
    # in one JSON artifact so plots can be reproduced without loading the model.
    metrics = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "mode": command,
        "labels": labels,
        "configuration": {
            "quantization": "4-bit NF4 with double quantization and BF16 compute",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "epochs": 1 if smoke else 2,
            "max_steps": 10 if smoke else -1,
            "learning_rate": 2e-4,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "max_length": MAX_LENGTH,
        },
        "parameters": {"trainable": trainable, "total": total},
        "baseline": baseline,
        "training": train_result.metrics,
        "validation": validation,
        "fine_tuned": fine_tuned,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    # Printing full reports makes an unattended terminal run self-contained;
    # the same information remains available in metrics.json for later plots.
    print(json.dumps({"baseline": baseline, "fine_tuned": fine_tuned}, indent=2))
    print(f"Saved adapter and metrics to {output_dir}")


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option.

    Batch size and gradient accumulation cannot be zero or negative. Performing
    this validation as an argparse ``type`` gives the user a normal command-line
    usage error before dataset/model loading begins.

    Args:
        value: Raw command-line token supplied by argparse.

    Returns:
        The parsed integer when it is at least one.

    Raises:
        ValueError: Indirectly from ``int`` when the token is not an integer.
        argparse.ArgumentTypeError: When the parsed integer is below one.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    """Parse the explicit command and dispatch to the requested workflow.

    Requiring one of four positional commands is the project's safety gate:
    running ``python train.py`` by itself only shows an argparse error and can
    never begin training. ``inspect`` avoids model loading, ``evaluate`` loads
    existing weights without training, and only ``smoke``/``run`` reach
    :func:`train`.

    Optional batch settings use :func:`positive_int` so invalid values fail at
    argument parsing. Gradient accumulation affects training only, but keeping
    it available for every command makes the CLI simple; it is ignored by the
    inspect/evaluate branches where it has no meaning.
    """
    parser = argparse.ArgumentParser(
        description="Inspect BANKING77 or explicitly start a QLoRA training run."
    )
    parser.add_argument("command", choices=("inspect", "evaluate", "smoke", "run"))
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--gradient-accumulation", type=positive_int, default=4)
    args = parser.parse_args()

    # Keep dispatch visibly explicit: a future command must be added to both the
    # choices and this branch instead of accidentally falling into training.
    if args.command == "inspect":
        inspect_dataset()
    elif args.command == "evaluate":
        evaluate_saved_models(args.batch_size)
    else:
        train(args.command, args.batch_size, args.gradient_accumulation)


if __name__ == "__main__":
    main()
