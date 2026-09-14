# Sudoku Deduction Evaluation

This evaluation measures `Qwen/Qwen3.5-2B` and `LiquidAI/LFM2.5-2.6B` on
structured Sudoku deductions. It does not load the BANKING77 LoRA adapter.

## Build The Evaluation Set

Build the Rust generator first:

```powershell
cd C:\Users\janhe\Projects\Finetune\sudoku-generator
cargo build --release
```

Generate candidates for the four deterministic solver-score bands. Keep the
metadata files; they are the source records for the immutable split.

```powershell
New-Item -ItemType Directory -Force C:\Users\janhe\Projects\Finetune\artifacts\sudoku-candidates | Out-Null
.\target\release\sudoku-generator.exe --count 300 --seed 1001 --min-score 1000 --max-score 1999 --output ..\artifacts\sudoku-candidates\b1.tsv --metadata ..\artifacts\sudoku-candidates\b1.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1002 --min-score 2000 --max-score 3999 --output ..\artifacts\sudoku-candidates\b2.tsv --metadata ..\artifacts\sudoku-candidates\b2.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1003 --min-score 4000 --max-score 7999 --output ..\artifacts\sudoku-candidates\b3.tsv --metadata ..\artifacts\sudoku-candidates\b3.jsonl
.\target\release\sudoku-generator.exe --count 300 --seed 1004 --min-score 8000 --output ..\artifacts\sudoku-candidates\b4.tsv --metadata ..\artifacts\sudoku-candidates\b4.jsonl
```

Select 250 puzzles from each band:

```powershell
cd C:\Users\janhe\Projects\Finetune
python evaluation\build_sudoku_eval.py `
  --input artifacts\sudoku-candidates\b1.jsonl `
  --input artifacts\sudoku-candidates\b2.jsonl `
  --input artifacts\sudoku-candidates\b3.jsonl `
  --input artifacts\sudoku-candidates\b4.jsonl `
  --generator-binary sudoku-generator\target\release\sudoku-generator.exe
```

The result is `artifacts/sudoku-eval-v1/`. Its manifest records hashes and
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

Adapters are saved to `artifacts/sudoku-deduction-qwen3.5-2b-qlora/` and
`artifacts/sudoku-deduction-lfm2.5-2.6b-qlora/`. Training uses completion-only
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
  --output artifacts\sudoku-traces.jsonl
```

Build balanced state-level examples and exclude the frozen evaluation puzzles:

```powershell
python evaluation\build_sudoku_deduction_dataset.py `
  --input artifacts\sudoku-traces.jsonl `
  --heldout artifacts\sudoku-eval-v1\eval-v1.metadata.jsonl `
  --output artifacts\sudoku-deductions.jsonl `
  --per-rule 1000
```

Each input state uses nine rows with fixed digits and bracketed candidate sets,
for example `r1: 5 3 [124] [26] 7 [248] ...`. The output target is a structured
deduction containing its rule, witness, placements, and/or candidate
eliminations. The solution is omitted; only its hash is retained.

Validate raw traces with `--trace` and validate the expanded examples without it:

```powershell
python evaluation\validate_sudoku_deduction_dataset.py artifacts\sudoku-traces.jsonl --trace
python evaluation\validate_sudoku_deduction_dataset.py artifacts\sudoku-deductions.jsonl
```

To build a benchmark-only set, use a new generator seed and exclude both the
solution-training examples and the frozen final-solution benchmark:

```powershell
python evaluation\build_sudoku_deduction_dataset.py `
  --input artifacts\sudoku-deduction-benchmark-traces-v1.jsonl `
  --heldout artifacts\sudoku-eval-v1\eval-v1.metadata.jsonl `
  --heldout artifacts\sudoku-deductions-v1.jsonl `
  --benchmark `
  --output artifacts\sudoku-deduction-benchmark-v1.jsonl `
  --per-rule 50
```

The first implementation emits naked/hidden subsets, locked candidates, and
basic fish sizes 2 through 8 in both orientations. Finned, sashimi, Franken,
Mutant, and Kraken fish are not implemented.

Evaluate a base model or adapter on the deduction benchmark:

```powershell
python evaluation\evaluate_sudoku_deductions.py `
  --dataset artifacts\sudoku-deduction-benchmark-v1.jsonl `
  --adapter artifacts\sudoku-deduction-qwen3.5-2b-qlora `
  --predictions artifacts\sudoku-deduction-benchmark-v1.qwen.predictions.jsonl `
  --metrics artifacts\sudoku-deduction-benchmark-v1.qwen.metrics.json `
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
  --dataset artifacts\sudoku-deduction-benchmark-v1.jsonl `
  --adapter artifacts\sudoku-deduction-lfm2.5-2.6b-qlora `
  --predictions artifacts\sudoku-deduction-benchmark-v1.lfm.predictions.jsonl `
  --metrics artifacts\sudoku-deduction-benchmark-v1.lfm.metrics.json `
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
python evaluation\evaluate_sudoku_deductions.py --dataset artifacts\sudoku-deduction-benchmark-v1.jsonl --adapter artifacts\sudoku-deduction-qwen3.5-2b-qlora --system-prompt-file evaluation\prompts\sudoku-deduction-v2.txt --predictions artifacts\sudoku-deduction-benchmark-v1.qwen-v2.predictions.jsonl --metrics artifacts\sudoku-deduction-benchmark-v1.qwen-v2.metrics.json --batch-size 4 --overwrite
```

This is a prompt-only ablation because the adapter was trained with the v1
prompt. Use the same v2 prompt when retraining if v2 is to be part of the final
model interface.

## Prompt Version 3 Ablation

V3 preserves the exact v1 JSON contract while adding concise rule definitions:

```powershell
python evaluation\evaluate_sudoku_deductions.py --dataset artifacts\sudoku-deduction-benchmark-v1.jsonl --adapter artifacts\sudoku-deduction-qwen3.5-2b-qlora --system-prompt-file evaluation\prompts\sudoku-deduction-v3.txt --predictions artifacts\sudoku-deduction-benchmark-v1.qwen-v3.predictions.jsonl --metrics artifacts\sudoku-deduction-benchmark-v1.qwen-v3.metrics.json --batch-size 4 --overwrite
```

This evaluates the existing v1-trained adapter with v3. Retraining is required
if v3 is intended to be the final model interface.

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
python evaluation\evaluate_sudoku_deductions.py --provider openrouter --dataset artifacts\sudoku-deduction-benchmark-v1.jsonl --system-prompt-file evaluation\prompts\sudoku-deduction-v3.txt --predictions artifacts\sudoku-deduction-benchmark-v1.openrouter.predictions.jsonl --metrics artifacts\sudoku-deduction-benchmark-v1.openrouter.metrics.json --batch-size 1 --overwrite
```
