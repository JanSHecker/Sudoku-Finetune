# QLoRA Banking Intent Router

This project fine-tunes
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
to classify English banking-support messages into the 77 intents in
[BANKING77](https://huggingface.co/datasets/PolyAI/banking77).

The code uses the parquet-based
[MTEB mirror](https://huggingface.co/datasets/mteb/banking77) because current
versions of `datasets` no longer execute the original dataset's loading script.
The original dataset is CC BY 4.0.

## Safety gate

Nothing happens when the modules are imported. Every operation requires an
explicit command:

```powershell
# Downloads and inspects the dataset. Does not load the model or use the GPU.
python train.py inspect

# Downloads the model and performs a 10-step GPU training check.
python train.py smoke

# Runs the complete baseline, two training epochs, and final evaluation.
python train.py run

# Re-evaluates the saved full adapter with macro and per-label metrics.
python train.py evaluate
```

Do not run `smoke` or `run` until the code and configuration have been reviewed.

The project uses the already-installed `torch`, `transformers`, `datasets`,
`peft`, `trl`, and `bitsandbytes` packages. No additional training framework is
needed.

## Data flow

`train.py` loads 9,993 training and 3,076 test rows. It makes a stratified,
seeded validation split of 1,000 rows from the training data. The original test
split is never used for training.

Each source row looks like this:

```text
text: I am still waiting on my card?
label_text: card_arrival
```

It becomes a conversational prompt-completion example:

```text
system: Classify the request. Valid labels: activate_my_card, ...
user: I am still waiting on my card?
assistant: card_arrival
```

All 77 labels are present in the system prompt. This makes the allowed output
space explicit for both the untouched base model and the fine-tuned model.
`completion_only_loss=True` means training loss is calculated only on the
assistant label, not on the repeated system prompt or user message.

## QLoRA configuration

The base model is loaded in 4-bit NF4 with double quantization. Its weights stay
frozen. LoRA adds small trainable matrices to every linear layer; those adapter
weights use BF16 computation.

| Setting | Value | Purpose |
| --- | ---: | --- |
| LoRA rank | 16 | Adapter capacity |
| LoRA alpha | 32 | Adapter update scale |
| LoRA dropout | 0.05 | Light regularization |
| Learning rate | 0.0002 | Typical adapter learning rate |
| Epochs | 2 | One focused full run |
| GPU batch size | 4 | Conservative default for 12 GB VRAM |
| Gradient accumulation | 4 | Effective training batch of 16 |
| Maximum length | 512 | Fits the label list and BANKING77 messages |

Four-bit weights alone occupy much less memory than the complete training
process. Activations, temporary CUDA buffers, and LoRA optimizer state also use
VRAM. If batch size 4 is too large, preserve the effective batch size with:

```powershell
python train.py smoke --batch-size 2 --gradient-accumulation 8
```

## Evaluation and artifacts

Before training, the quantized base model classifies the selected test set with
deterministic generation. The same evaluator runs after training. It records:

- exact-label accuracy;
- the percentage of outputs that are one of the 77 valid labels;
- macro precision, recall, and F1 across all labels;
- support, precision, recall, and F1 for every label;
- the ten most common incorrect expected/predicted pairs;
- training and validation metrics, runtime, parameter counts, and peak VRAM.

The full run saves only the LoRA adapter, tokenizer files, and `metrics.json` in
`artifacts/qwen2.5-1.5b-banking77-qlora`. The smoke run uses a separate
`-smoke` directory.

After a completed full run, classify one message with:

```powershell
python predict.py "My cash withdrawal is still pending"
```

`predict.py` loads the frozen base model plus the saved adapter. It rejects
outputs that are not valid BANKING77 labels.

## Sudoku Deduction QLoRA

The Sudoku trainer is separate from the BANKING77 workflow. It trains on
state-level structured deductions rather than attempting to emit a completed
grid in one shot. The dataset preserves puzzle-level train, validation, and
test splits; the separate deduction benchmark is never read during training:

```powershell
python train_sudoku.py inspect
python train_sudoku.py smoke --model-id Qwen/Qwen3.5-2B --batch-size 1 --gradient-accumulation 1
python train_sudoku.py run --model-id Qwen/Qwen3.5-2B --batch-size 1 --gradient-accumulation 8
```

Train the LFM challenger into its separate adapter directory:

```powershell
python train_sudoku.py smoke --model-id LiquidAI/LFM2.5-2.6B --batch-size 1 --gradient-accumulation 1
python train_sudoku.py run --model-id LiquidAI/LFM2.5-2.6B --batch-size 1 --gradient-accumulation 8
```

Evaluate either adapter with `evaluation\evaluate_sudoku_deductions.py` on
`artifacts\sudoku-deduction-benchmark-v1.jsonl`. See
`evaluation\README.md` for the complete evaluation commands.
