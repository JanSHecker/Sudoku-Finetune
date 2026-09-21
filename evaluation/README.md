# Sudoku Deduction Evaluation

This evaluation measures `Qwen/Qwen3.5-2B` and `LiquidAI/LFM2.5-2.6B` on
structured Sudoku deductions.

## Build The Evaluation Set

Build the Rust generator first:

```powershell
cd C:\Users\janhe\Projects\Finetune\sudoku-generator
cargo build --release
```

Generate candidates for the four deterministic solver-score bands. Keep the
metadata files; they are the source records for the immutable split.

```powershell
New-Item -ItemType Directory -Force C:\Users\janhe\Projects\Finetune\artifacts\datasets\training\candidates | Out-Null
.\target\release\sudoku-generator.exe --count 300 --seed 1001 --min-score 1000 --max-score 1999 --output ..\artifacts\datasets\training\candidates\b1.tsv --metadata ..\artifacts\datasets\training\candidates\b1.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1002 --min-score 2000 --max-score 3999 --output ..\artifacts\datasets\training\candidates\b2.tsv --metadata ..\artifacts\datasets\training\candidates\b2.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1003 --min-score 4000 --max-score 7999 --output ..\artifacts\datasets\training\candidates\b3.tsv --metadata ..\artifacts\datasets\training\candidates\b3.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1004 --min-score 8000 --output ..\artifacts\datasets\training\candidates\b4.tsv --metadata ..\artifacts\datasets\training\candidates\b4.jsonl
```

Select 250 puzzles from each band:

```powershell
cd C:\Users\janhe\Projects\Finetune
python evaluation\build_sudoku_eval.py `
  --input artifacts\datasets\training\candidates\b1.jsonl `
  --input artifacts\datasets\training\candidates\b2.jsonl `
  --input artifacts\datasets\training\candidates\b3.jsonl `
  --input artifacts\datasets\training\candidates\b4.jsonl `
  --generator-binary sudoku-generator\target\release\sudoku-generator.exe
```

The result is `artifacts/datasets/validation/sudoku-eval-v1/`. Its manifest records hashes and
selection parameters. Do not use this set for fine-tuning.

## Smoke Evaluation

Run ten examples before the full run:

```powershell
python evaluation\evaluate_sudoku.py --limit 10 --overwrite
```

The default `direct` prompt is intentionally concise. To test whether explicit
rule explanations help the base model, run the alternate protocol:

```powershell
python evaluation\evaluate_sudoku.py --prompt rules --limit 100 --overwrite
```

The prompt variant is recorded in every prediction and metrics artifact, so
the two results must not be combined.

The first run may resolve the pinned base-model snapshot from the Hugging Face
cache. Add `--local-files-only` to require that all files are already cached.

## Full Baseline

```powershell
python evaluation\evaluate_sudoku.py --batch-size 4 --overwrite
```

If interrupted, continue without repeating completed examples:

```powershell
python evaluation\evaluate_sudoku.py --batch-size 4 --resume
```

The evaluator writes:

- `predictions.base-4bit.jsonl`: raw output, token IDs, parsed output, validity checks, and answer key per example;
- `metrics.base-4bit.json`: overall and per-score-band metrics with a 95% Wilson interval;
- the frozen `eval-v1.tsv`, metadata, and manifest from the set-building step.

## Deduction QLoRA Training

The final-grid one-shot trainer has been replaced. The trainer reads the
state-level deduction dataset, uses its puzzle-level `train` and `validation`
splits, and never reads the held-out deduction benchmark. Inspect the split:

```powershell
python train_sudoku.py inspect
```

Run a short GPU smoke test for each model before the full job:

```powershell
python train_sudoku.py smoke --model-id Qwen/Qwen3.5-2B --batch-size 1 --gradient-accumulation 1
python train_sudoku.py smoke --model-id LiquidAI/LFM2.5-2.6B --batch-size 1 --gradient-accumulation 1
```

Run the three-epoch training jobs:

```powershell
python train_sudoku.py run `
  --model-id Qwen/Qwen3.5-2B `
  --batch-size 1 `
  --gradient-accumulation 8

python train_sudoku.py run `
  --model-id LiquidAI/LFM2.5-2.6B `
  --batch-size 1 `
  --gradient-accumulation 8
```

Adapters are saved to `artifacts/checkpoints/deduction/sudoku-deduction-qwen3.5-2b-qlora/` and
`artifacts/checkpoints/deduction/sudoku-deduction-lfm2.5-2.6b-qlora/`. Training uses completion-only
loss against the structured JSON deduction. LFM's mandatory `<think>` block is
closed before the JSON target so its training format matches the evaluator.

## Deduction Dataset

The public `sudoku-generator/` crate emits uniquely solvable puzzle TSV and
metadata. The `--trace` commands below document the historical trace-capable
generator used to produce the existing experiment artifacts; that larger
deduction engine is not included in this checkout. The generated artifacts are
ignored from Git, so use the preserved metadata and reports only as experiment
context unless a trace-capable generator is supplied.

The Rust generator can emit replayable logical traces. The trace format includes
the complete candidate state before and after each deduction and all deductions
available at that state:

```powershell
sudoku-generator\target\release\sudoku-generator.exe `
  --trace `
  --count 500 `
  --seed 9001 `
  --min-score 1000 `
  --output artifacts\datasets\training\traces\sudoku-traces.jsonl
```

Build balanced state-level examples and exclude the frozen evaluation puzzles:

```powershell
python evaluation\build_sudoku_deduction_dataset.py `
  --input artifacts\datasets\training\traces\sudoku-traces.jsonl `
  --heldout artifacts\datasets\validation\sudoku-eval-v1\eval-v1.metadata.jsonl `
  --output artifacts\datasets\training\deduction\sudoku-deductions.jsonl `
  --per-rule 1000
```

Each input state uses nine rows with fixed digits and bracketed candidate sets,
for example `r1: 5 3 [124] [26] 7 [248] ...`. The output target is a structured
deduction containing its rule, witness, placements, and/or candidate
eliminations. The solution is omitted; only its hash is retained.

Validate raw traces with `--trace` and validate the expanded examples without it:

```powershell
python evaluation\validate_sudoku_deduction_dataset.py artifacts\datasets\training\traces\sudoku-traces.jsonl --trace
python evaluation\validate_sudoku_deduction_dataset.py artifacts\datasets\training\deduction\sudoku-deductions.jsonl
```

To build a benchmark-only set, use a new generator seed and exclude both the
solution-training examples and the frozen final-solution benchmark:

```powershell
python evaluation\build_sudoku_deduction_dataset.py `
  --input artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-traces-v1.jsonl `
  --heldout artifacts\datasets\validation\sudoku-eval-v1\eval-v1.metadata.jsonl `
  --heldout artifacts\datasets\training\deduction\sudoku-deductions-v1.jsonl `
  --benchmark `
  --output artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl `
  --per-rule 50
```

The first implementation emits naked/hidden subsets, locked candidates, and
basic fish sizes 2 through 8 in both orientations. Finned, sashimi, Franken,
Mutant, and Kraken fish are not implemented.

Evaluate a base model or adapter on the deduction benchmark:

```powershell
python evaluation\evaluate_sudoku_deductions.py `
  --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl `
  --adapter artifacts\checkpoints\deduction\sudoku-deduction-qwen3.5-2b-qlora `
  --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen.predictions.jsonl `
  --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen.metrics.json `
  --batch-size 4 `
  --overwrite
```

The evaluator reports JSON parse rate, independently sound deduction rate,
membership in the complete valid-deduction set, rule-family accuracy, and exact
match to the selected target. It uses a separate system prompt and does not use
the final-solution evaluator.

Evaluate LFM with the same frozen benchmark and a separate output file:

```powershell
python evaluation\evaluate_sudoku_deductions.py `
  --model-id LiquidAI/LFM2.5-2.6B `
  --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl `
  --adapter artifacts\checkpoints\deduction\sudoku-deduction-lfm2.5-2.6b-qlora `
  --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.lfm.predictions.jsonl `
  --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.lfm.metrics.json `
  --batch-size 4 `
  --overwrite
```

LFM reasoning output is removed before JSON scoring. An untrained LFM
checkpoint may need a larger `--max-new-tokens` budget to reach its answer.

## Prompt Version 2 Ablation

The v2 prompt adds formal Sudoku rules, coordinate conventions, technique
definitions, witness requirements, and the JSON contract. Evaluate the existing
Qwen adapter with it using a separate artifact name:

```powershell
python evaluation\evaluate_sudoku_deductions.py --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl --adapter artifacts\checkpoints\deduction\sudoku-deduction-qwen3.5-2b-qlora --system-prompt-file evaluation\prompts\sudoku-deduction-v2.txt --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen-v2.predictions.jsonl --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen-v2.metrics.json --batch-size 4 --overwrite
```

This is a prompt-only ablation because the adapter was trained with the v1
prompt. Use the same v2 prompt when retraining if v2 is to be part of the final
model interface.

## Prompt Version 3 Ablation

V3 preserves the exact v1 JSON contract while adding concise rule definitions:

```powershell
python evaluation\evaluate_sudoku_deductions.py --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl --adapter artifacts\checkpoints\deduction\sudoku-deduction-qwen3.5-2b-qlora --system-prompt-file evaluation\prompts\sudoku-deduction-v3.txt --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen-v3.predictions.jsonl --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.qwen-v3.metrics.json --batch-size 4 --overwrite
```

This evaluates the existing v1-trained adapter with v3. Retraining is required
if v3 is intended to be the final model interface.

## Phase 3 Representation Warm-Up

Phase 3 is a separate training process. It creates auxiliary examples from the
existing state-level deduction dataset and trains a separate adapter. The
earlier deduction trainer and its artifacts remain unchanged.

Build the representation dataset from the preserved training, validation, and
test state splits. The builder deduplicates repeated states and emits eight
tasks per state by default:

```powershell
python evaluation\build_sudoku_representation_dataset.py `
  --input artifacts\datasets\training\deduction\sudoku-deductions-v1.jsonl `
  --output artifacts\datasets\training\representation\sudoku-representation-v1.jsonl `
  --report artifacts\results\validation\representation\sudoku-representation-v1.report.json `
  --tasks-per-state 8
```

The generated Phase 3 dataset contains only visible board and candidate-grid
information. It must not be built from the held-out benchmark or benchmark
predictions.

Inspect and smoke-test the isolated trainer before a full run:

```powershell
python train_sudoku_representation.py inspect `
  --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl

python train_sudoku_representation.py smoke `
  --model-id Qwen/Qwen3.5-2B `
  --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl `
  --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora `
  --batch-size 1 `
  --gradient-accumulation 1
```

Run the one-epoch warm-up with a separate artifact directory:

```powershell
python train_sudoku_representation.py run `
  --model-id Qwen/Qwen3.5-2B `
  --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl `
  --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora `
  --batch-size 1 `
  --gradient-accumulation 8 `
  --save-steps 10
```

The trainer saves full resumable checkpoints every 10 optimizer steps and keeps
the three most recent checkpoints. If Windows restarts or the process fails,
continue the same run with the same training arguments and `--resume`:

```powershell
python train_sudoku_representation.py run `
  --model-id Qwen/Qwen3.5-2B `
  --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl `
  --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora `
  --batch-size 1 `
  --gradient-accumulation 8 `
  --save-steps 10 `
  --resume
```

The run directory contains `run_config.json`, `run_status.json`, and
`checkpoint-*` directories. Resume restores the model, optimizer, scheduler,
trainer progress, and random-number state. The final adapter and metrics are
written only after training and evaluation complete, but checkpoints remain
usable if finalization is interrupted.

The Phase 3 adapter is not automatically used by the old deduction pipeline.
The existing `train_sudoku.py` supports an additive `--initial-adapter` option
for a later deduction run, but the deduction examples should first be rebuilt
with the same explicit box-separated representation.

## Phase 3 Representation Dataset V2

V2 keeps the v1 system prompt and task wording unchanged. It changes only the
query sampler: coordinate and unit queries cover the board structure
deterministically, peer queries cover four relation classes, and candidate
location queries are balanced between empty and non-empty results. Candidate
locations are also reported by expected result size.

Build and inspect the v2 artifact:

```powershell
python evaluation\build_sudoku_representation_dataset.py `
  --input artifacts\datasets\training\deduction\sudoku-deductions-v1.jsonl `
  --output artifacts\datasets\training\representation\sudoku-representation-v2.jsonl `
  --report artifacts\results\validation\representation\sudoku-representation-v2.report.json `
  --tasks-per-state 8 `
  --format-version 2

python train_sudoku_representation.py inspect `
  --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl
```

Start a corrective run from the existing representation adapter in a new
directory. `--initial-adapter` starts a new optimizer run; it is distinct from
`--resume`, which restores a checkpoint and optimizer state:

```powershell
python train_sudoku_representation.py run `
  --model-id Qwen/Qwen3.5-2B `
  --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl `
  --initial-adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora `
  --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2 `
  --batch-size 1 `
  --gradient-accumulation 8 `
  --learning-rate 0.0001 `
  --max-steps 400 `
  --save-steps 10
```

The first recorded v2 scratch phase resumed from `checkpoint-635` and completed
at `checkpoint-952` using batch size 2 and gradient accumulation 4. This was
approximately 60% of one epoch. The artifact directory is named
`sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct` because that was its
original staged-run label. The exact run metadata and validation result are
recorded in its local `metrics.json` and summarized in the project `README.md`.

To continue that run to approximately one full epoch, resume the latest
checkpoint and set the total step target to 1,586:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --batch-size 2 --gradient-accumulation 4 --learning-rate 0.0002 --max-steps 1586 --save-steps 10 --save-total-limit 3 --resume
```

That continuation completed at `checkpoint-1586` and reached one full epoch.
The final validation loss was `0.00367` with `99.86%` teacher-forced token
accuracy. Free-generation evaluation on the 1,472-example test split achieved
`95.52%` exact answer accuracy overall and `96.97%` equal-weight task-macro
accuracy. The recorded evaluation artifacts are
`artifacts\results\validation\representation\sudoku-representation-v2.scratch-20pct.test.metrics.json` and
`artifacts\results\validation\representation\sudoku-representation-v2.scratch-20pct.test.predictions.jsonl`.

## Candidate-Location Augmentation

The remaining representation errors are concentrated in candidate-location
enumeration, especially non-empty box queries and multi-cell results. A larger
12-query-per-state exploratory artifact exists, but the light continuation
should use four additional queries per state:

```powershell
python evaluation\build_candidate_location_augmentation.py --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output artifacts\datasets\training\representation\sudoku-candidate-location-augmentation-lite-v1.train.jsonl --report artifacts\results\validation\representation\sudoku-candidate-location-augmentation-lite-v1.train.report.json --queries-per-state 4 --seed 42 --split train
```

The builder preserves the existing train/validation puzzle split, excludes all
184 test states, and emits 6,344 train-only augmentation examples. Combined
with the original v2 dataset, this is 19,032 train and the unchanged 2,016-row
v2 validation set. It balances row, column, and box units and empty, one, few,
and many result sizes; non-empty box categories receive extra weight. A
two-split light artifact can also be built for targeted validation diagnostics,
but it is not the historical comparison protocol.

Use the original v2 dataset as replay data when continuing from the completed
adapter. The proposed short run uses a new output directory and a lower
learning rate:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --input artifacts\datasets\training\representation\sudoku-candidate-location-augmentation-lite-v1.train.jsonl --initial-adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1 --batch-size 2 --gradient-accumulation 4 --learning-rate 0.00005 --max-steps 400 --save-steps 10 --save-total-limit 3
```

The completed candidate-lite run reached `98.44%` overall exact accuracy and
`94.02%` candidate-location exact accuracy on the unchanged v2 test split.
Non-empty candidate precision/recall reached `98.62%` / `98.19%` by mean and
`98.11%` / `97.74%` by micro averaging. Results are stored in
`artifacts\results\validation\representation\sudoku-representation-v2.candidate-lite.test.metrics.json` and
`artifacts\results\validation\representation\sudoku-representation-v2.candidate-lite.test.predictions.jsonl`.

Evaluate the corrective adapter on the unchanged v2 test protocol:

```powershell
python evaluation\evaluate_sudoku_representation.py `
  --dataset artifacts\datasets\training\representation\sudoku-representation-v2.jsonl `
  --adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1 `
  --split test `
  --predictions artifacts\results\validation\representation\sudoku-representation-v2.candidate-lite.test.predictions.jsonl `
  --metrics artifacts\results\validation\representation\sudoku-representation-v2.candidate-lite.test.metrics.json `
  --batch-size 4 `
  --overwrite
```

## Deduction Prompt Format V2

The representation adapter expects the labeled, box-separated grid used by the
v2 representation dataset. Convert the old deduction and benchmark artifacts
into separate v2 prompt-format copies; the v1 files are not modified:

```powershell
python evaluation\format_sudoku_deduction_prompts.py --input artifacts\datasets\training\deduction\sudoku-deductions-v1.jsonl --output artifacts\datasets\training\deduction\sudoku-deductions-v2.jsonl --report artifacts\results\validation\deduction\sudoku-deductions-v2.prompt-format.report.json

python evaluation\format_sudoku_deduction_prompts.py --input artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl --output artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v2.jsonl --report artifacts\results\validation\deduction\sudoku-deduction-benchmark-v2.prompt-format.report.json
```

Both converted files pass logical dataset validation and preserve the original
IDs, targets, splits, and puzzle sets. Smoke-test the representation warm-start
before a full deduction run:

```powershell
python train_sudoku.py smoke --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\deduction\sudoku-deductions-v2.jsonl --initial-adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1 --batch-size 2 --gradient-accumulation 4
```

For the eventual benchmark, evaluate against the matching v2 prompt-format
benchmark:

```powershell
python evaluation\evaluate_sudoku_deductions.py --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v2.jsonl --adapter artifacts\checkpoints\deduction\sudoku-deduction-qwen3.5-2b-representation-warmstart-v2 --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v2.warmstart.predictions.jsonl --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v2.warmstart.metrics.json --batch-size 4 --overwrite
```

For a staged experiment, keep the full dataset and limit optimizer steps so a
later resume sees the same sampler and can continue the Trainer state. With the
current 12,688 training rows and gradient accumulation of 8, 160 steps is
approximately 10% of one epoch:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora --batch-size 1 --gradient-accumulation 8 --epochs 1 --max-steps 160 --save-steps 10 --save-total-limit 3
```

After reviewing the validation result, continue the same epoch with:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v1.jsonl --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora --batch-size 1 --gradient-accumulation 8 --epochs 1 --max-steps -1 --save-steps 10 --save-total-limit 3 --resume
```

## Representation Generation Benchmark

Training metrics use teacher forcing. Evaluate the saved adapter by free
generation on the held-out representation test split:

```powershell
python evaluation\evaluate_sudoku_representation.py --dataset artifacts\datasets\training\representation\sudoku-representation-v1.jsonl --adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora --split test --predictions artifacts\results\validation\representation\sudoku-representation-v1.test.predictions.jsonl --metrics artifacts\results\validation\representation\sudoku-representation-v1.test.metrics.json --batch-size 4 --overwrite
```

The benchmark reports JSON parse rate, schema rate, task-label rate, exact
answer rate, per-task sample counts, majority baselines, Wilson 95% intervals,
and an equal-weight task macro average. Candidate-location and naked-single
scan results additionally report balanced empty/non-empty accuracy and
item-list precision/recall. Peer and coordinate tasks report field-level
accuracy and baselines. It writes each raw prediction immediately; if
interrupted, rerun the same command with `--resume` instead of `--overwrite`.

## OpenRouter External Model

The same evaluator can call an external model through OpenRouter instead of
loading a local checkpoint. Put the API key and model name in the project-root
`.env` file. The default model is `z-ai/glm-5.3-flash`; change `OPENROUTER_MODEL` if a
different current SOTA model is preferred. `.env` is ignored by Git and the key
is never written to prediction or metrics artifacts.

OpenRouter evaluation makes one request per example, so use `--batch-size 1`.
It is resumable with `--resume`; use `--overwrite` for a fresh run. This is a
practical external-model ceiling for the exact prompt and benchmark, not a
proof that the model has internalized the Sudoku algorithm.

```powershell
python evaluation\evaluate_sudoku_deductions.py --provider openrouter --dataset artifacts\datasets\validation\deduction\sudoku-deduction-benchmark-v1.jsonl --system-prompt-file evaluation\prompts\sudoku-deduction-v3.txt --predictions artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.openrouter.predictions.jsonl --metrics artifacts\results\validation\deduction\sudoku-deduction-benchmark-v1.openrouter.metrics.json --batch-size 1 --overwrite
```
