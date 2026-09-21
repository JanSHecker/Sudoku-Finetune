# Sudoku Fine-Tuning Experiment

Status: the training pipeline and logical validator work, but the current
language model does not generalize Sudoku deduction reliably.

## Objective

The experiment started by testing whether a small language model could solve a
Sudoku by producing a completed 9x9 grid. That one-shot approach was later
abandoned. The current objective is narrower and more verifiable: produce one
sound, structured deduction from a Sudoku state containing its candidate grid.

The model must output JSON describing:

- the deduction technique;
- its witness;
- placements, if any;
- candidate eliminations, if any.

The evaluator independently checks the witness and state transition. It accepts
any sound deduction available in the state, not only the selected training
target.

## Artifact Layout

Generated artifacts are ignored by Git and are organized by role:

- `artifacts/datasets/training/`: datasets used to train or generate training data.
- `artifacts/datasets/validation/`: held-out datasets and immutable evaluation inputs.
- `artifacts/checkpoints/`: model adapters and training checkpoints, grouped by task.
- `artifacts/results/validation/`: validation predictions, metrics, reports, and checks.

The task-specific folders below those categories are `deduction`,
`representation`, `solving`, `candidates`, and `traces` where applicable.

## Hardware And Configuration

- GPU: NVIDIA RTX 4070 with 12 GB VRAM.
- Quantization: 4-bit NF4 with double quantization.
- Compute: BF16.
- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.05.
- LoRA target modules: `all-linear`.
- Learning rate: `0.0002`.
- Effective batch size in full deduction training: 8.
- Maximum sequence length: 1024.
- Base models:
  - `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
  - `LiquidAI/LFM2.5-2.6B`, revision `654f9463ce32b05d0429d76fe1f580b27d4c1ac0`.

## Earlier One-Shot Approach

The first Sudoku trainer asked the model to emit the complete solved grid with
no explanation. This established that QLoRA training could run on the local
GPU, but it was not the desired deduction behavior.

### Qwen2.5 full one-shot run

Artifact: `artifacts/checkpoints/solving/sudoku-qwen2.5-1.5b-qlora/metrics.json`

- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Training examples: 12,000.
- Training epochs: 3.
- Training loss: `0.1851`.
- Validation loss: `0.1566`.
- Validation teacher-forced token accuracy: `92.05%`.
- Training runtime: approximately 133 minutes.

These numbers show successful optimization against completed-grid targets. They
do not establish generated Sudoku-solving accuracy, and the one-shot approach
was abandoned in favor of state-level deductions.

### Qwen3.5 and LFM one-shot smoke runs

The newer models also loaded and trained successfully in the old one-shot
pipeline:

| Model | Train speed | Train loss | Validation loss | Validation token accuracy |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | 0.374 samples/s | 0.895 | 0.687 | 72.17% |
| LFM2.5-2.6B | 1.444 samples/s | 3.608 | 3.154 | 35.12% |

LFM was approximately four times faster by sample in these short runs. Its
architecture is optimized for efficient inference, while Qwen was also using a
slower PyTorch fallback because the optional linear-attention kernels were not
installed.

## Sudoku Generator And Data

The deterministic Rust puzzle generator is included in `sudoku-generator/`.
It creates uniquely solvable puzzles and records a reproducible solver-search
difficulty score. The hidden solution is used for validation during generation
and is omitted from model examples except for a SHA-256 hash.

Build and run it with:

```powershell
cd sudoku-generator
cargo run --release -- --count 10 --seed 42 --min-score 1000 --output puzzles.tsv
```

The larger trace artifacts described below were generated during the original
experiment and are intentionally ignored from the public source repository.

The historical trace-capable engine used for the results below supported:

- naked and hidden singles;
- naked and hidden pairs, triples, and quads;
- pointing and claiming;
- basic fish sizes 2 through 8 in row and column orientations.

Finned, sashimi, Franken, Mutant, and Kraken fish are not implemented.

### Training dataset

Artifact: `artifacts/datasets/training/deduction/sudoku-deductions-v1.jsonl`

- Source records: 500 traces.
- Source puzzles: 452.
- Expanded candidate examples before balancing: 707,677.
- Selected examples: 2,143.
- Train rows: 1,678 from 357 puzzles.
- Validation rows: 268 from 53 puzzles.
- Test rows: 197 from 42 puzzles.
- All train, validation, and test puzzle sets are disjoint.

The independent validator confirms all 2,143 examples are sound.

### Held-out benchmark

Artifact: `artifacts/datasets/validation/deduction/sudoku-deduction-benchmark-v1.jsonl`

- Benchmark rows: 1,094.
- Benchmark puzzles: 401.
- 50 examples per rule in most rule categories.
- The benchmark uses separate puzzles and is not read by the trainer.
- The evaluator uses independent witness, candidate, and transition checks.

## Deduction Trainer

`train_sudoku.py` was replaced with a deduction-specific trainer. It:

- reads `artifacts/datasets/training/deduction/sudoku-deductions-v1.jsonl` by default;
- trains only on rows marked `train`;
- validates only on rows marked `validation`;
- excludes the `test` split from training;
- rejects malformed examples and conflicting split puzzles;
- applies completion-only loss to the JSON deduction;
- supports Qwen3.5 and LFM chat templates;
- handles LFM's mandatory `<think>` block;
- saves separate adapters for each model.

The smoke adapters are:

- `artifacts/checkpoints/deduction/sudoku-deduction-qwen3.5-2b-qlora-smoke`;
- `artifacts/checkpoints/deduction/sudoku-deduction-lfm2.5-2.6b-qlora-smoke`.

Both smoke runs completed successfully. The full Qwen adapter is:

- `artifacts/checkpoints/deduction/sudoku-deduction-qwen3.5-2b-qlora`.

## Deduction Training Results

### Smoke training

Both models completed a 10-step smoke run and their adapters loaded through the
deduction evaluator.

| Model | Train loss | Validation loss | Validation token accuracy | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | 1.257 | 0.740 | 76.31% | 51.8 s |
| LFM2.5-2.6B | 1.706 | 0.930 | 77.65% | 15.5 s |

The smoke scores only verify the end-to-end pipeline. They are not meaningful
task-quality measurements after ten optimizer steps.

### Full Qwen deduction run

Artifact: `artifacts/checkpoints/deduction/sudoku-deduction-qwen3.5-2b-qlora/metrics.json`

- Training epochs: 3.
- Training loss: `0.2638`.
- Validation loss: `0.2278`.
- Validation teacher-forced token accuracy: `91.10%`.
- Training plus validation runtime: approximately 49 minutes.

The low validation loss and high token accuracy show that the adapter learned
the target JSON format and many token patterns. They do not show that the model
can generate valid deductions freely.

## Held-Out Qwen Evaluation

Artifact: `artifacts/results/validation/deduction/sudoku-deduction-benchmark-v1.qwen.metrics.json`

The adapter was evaluated on all 1,094 held-out benchmark examples using
deterministic generation.

| Metric | Result |
| --- | ---: |
| JSON parse rate | 100.00% |
| Sound deduction rate | 0.457%: 5/1,094 |
| Valid deduction rate | 0.457%: 5/1,094 |
| Valid rule rate | 0.457%: 5/1,094 |
| Exact selected target rate | 0.091%: 1/1,094 |
| Mean generation time | 10.70 s/example |

This is a clear partial success and a clear task failure:

- Success: the model always produced parseable JSON in the expected shape.
- Failure: almost every witness, placement, or elimination was logically wrong.
- The model sometimes produced plausible fish structures with incorrect sizes,
  orientations, cells, or eliminations.
- Some valid outputs used a different deduction from the selected target, which
  is correctly accepted by the evaluator.

The adapter therefore learned formatting much better than Sudoku reasoning.

## Prompt Findings

The current system prompt is:

```text
You are a Sudoku deduction engine. Identify one logically sound deduction from
the supplied state. Do not guess or provide a final solution. Return only the
requested JSON deduction.
```

It defines the role and output intent, but it does not define Sudoku constraints,
techniques, witnesses, or cell indexing. The dataset gives examples of those
fields, but does not provide an explicit rulebook or natural-language
explanation.

Adding a rulebook to the system prompt would be task specification rather than
benchmark leakage, provided the same prompt is used for the base and fine-tuned
models and it contains no puzzle-specific answers.

The first rulebook prompt is stored at
`evaluation/prompts/sudoku-deduction-v2.txt`. It adds Sudoku constraints,
zero-based cell and unit coordinates, definitions for every supported technique,
witness requirements, and the exact JSON contract. The evaluator supports this
as an explicit prompt-file ablation without changing the built-in v1 prompt or
the existing v1 metrics.

The first one-example v2 wiring check produced parseable output and used the
full prompt successfully. The existing Qwen adapter was trained with v1, so a
full v2 run measures the effect of providing the rulebook at evaluation time;
it is not a v2-trained adapter result.

### Full v2 prompt ablation

Artifact: `artifacts/results/validation/deduction/sudoku-deduction-benchmark-v1.qwen-v2.metrics.json`

The existing v1-trained Qwen adapter was evaluated on the same 1,094 examples
with the v2 rulebook prompt:

| Metric | v1 prompt | v2 prompt |
| --- | ---: | ---: |
| JSON parse rate | 100.00% | 77.51% |
| Sound deduction rate | 0.457% | 0.000% |
| Valid deduction rate | 0.457% | 0.000% |
| Valid rule rate | 0.457% | 0.000% |
| Exact target rate | 0.091% | 0.000% |
| Mean generation time | 10.70 s | 9.87 s |

The longer prompt did not transfer usable rule knowledge to the existing
adapter. It instead caused schema drift: outputs omitted technique fields,
omitted fish sizes, or used malformed witness nesting. The v2 prompt should not
be used as a final interface without retraining and re-evaluating the model
with the same prompt.

### V3 prompt

The corrected prompt is stored at
`evaluation/prompts/sudoku-deduction-v3.txt`. It preserves the v1 contract
explicitly: all four top-level keys are required, placements and eliminations
are always arrays, fish size and orientation are required, and every witness
shape is specified using the evaluator's nested fields. It is shorter than v2
and keeps the rulebook concise.

A one-example wiring check completed successfully with the existing adapter.
The full evaluation command is documented in `evaluation/README.md`. The
existing adapter was trained with v1, so this remains a prompt-only ablation;
v3 must also be used during retraining for a final v3 experiment.

The evaluator also supports an external OpenRouter provider. Its request path
uses the same benchmark rows, v3 system prompt, JSON parser, independent
soundness validator, and metrics summary. The API key is read from the
project-root `.env` file and is not recorded in artifacts. The external run is
intended to estimate a practical ceiling for this prompt and benchmark; it is
not directly comparable to local inference speed or deterministic decoding.

## Current Conclusion

The project has a working and independently validated data pipeline, QLoRA
training pipeline, model adapters, and benchmark evaluator. The current model
result should not be described as successful Sudoku deduction. It should be
described as successful structured-output fine-tuning with very poor logical
generalization.

The next experiment should not simply add more epochs to the unchanged setup.
The recommended sequence is:

1. Evaluate the unfine-tuned base model on the same benchmark.
2. Add a compact formal rulebook and coordinate specification to the prompt.
3. Start with singles and pairs before training triples, quads, locked
   candidates, and fish.
4. Use generated validation soundness, rather than teacher-forced token accuracy,
   as the primary training signal.
5. Consider having the model propose a deduction that the deterministic Sudoku
   engine verifies before accepting it.

## Phase 3: Representation Warm-Up

Phase 3 is an isolated auxiliary-training process. It does not replace or
rewrite the earlier one-shot, v1, v2, or v3 pipelines. The builder creates
examples from the visible board and candidate grid only, with no deductions or
hidden solution values in the targets.

The representation tasks are:

- cell and candidate lookup;
- coordinate-to-cell, row, column, and box conversion;
- listing the cells in a row, column, or box;
- locating a digit in a unit;
- peer and unit relationships;
- scanning for naked singles.

The grid is rendered with row and column labels and explicit 3x3 box
boundaries. The warm-up adapter is saved separately under
`artifacts/checkpoints/representation/sudoku-representation-qwen3.5-2b-qlora/`. The existing deduction
trainer accepts it later through its additive `--initial-adapter` option; a
later deduction dataset must use the same representation format for that
continuation to be meaningful.

The preserved Phase 3 commands are documented in `evaluation/README.md`.

The no-prompt-change Phase 3 follow-up dataset is
`artifacts/datasets/training/representation/sudoku-representation-v2.jsonl`. It keeps the original task wording
while using deterministic box/peer coverage and balanced candidate-location
queries. The representation trainer can continue from the existing adapter
with its additive `--initial-adapter` option.

## Experiment History

This section is the durable summary of completed experiment runs. Detailed
artifacts remain in `artifacts/`, but that directory is ignored by Git.

### 2026-09-17: Representation V2 staged run

Artifact: `artifacts/checkpoints/representation/sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct/`

- Dataset: `artifacts/datasets/training/representation/sudoku-representation-v2.jsonl`.
- Dataset fingerprint: `c7956992a2e4de2668ab9f06a87491c5ad0ef34eec9e1aad571b8c89770c86b7`.
- Model: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Dataset usage: 12,688 training rows and 2,016 validation rows.
- Configuration: batch size 2, gradient accumulation 4, effective batch size 8,
  learning rate `0.0002`, BF16 4-bit QLoRA.
- Checkpoint transition: resumed from `checkpoint-635` and completed at
  `checkpoint-952`.
- Progress: 952 of approximately 1,586 one-epoch optimizer steps, or 60.03%.
- Training loss: `0.00413`.
- Validation loss: `0.00716`.
- Validation teacher-forced token accuracy: `99.73%`.
- Status: completed as the first staged phase at step 952; the run was later
  continued from this checkpoint.

Recorded command:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --batch-size 2 --gradient-accumulation 4 --learning-rate 0.0002 --max-steps 952 --save-steps 10 --resume
```

The artifact directory retains the checkpoints from this staged phase. Its name
includes `20pct` because that was the original staged-run label.

### 2026-09-18: Representation V2 full-epoch continuation

The same v2 run continued from `checkpoint-952` to `checkpoint-1586`, completing
one epoch with the unchanged dataset and optimizer configuration.

- Final artifact: `artifacts/checkpoints/representation/sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct/`.
- Final checkpoint: `checkpoint-1586`.
- Training loss: `0.00382`.
- Validation loss: `0.00367`.
- Validation teacher-forced token accuracy: `99.86%`.
- Final training epoch: `1.0`.
- Continuation runtime: approximately 2 hours 23 minutes.
- Status: completed.

Held-out free-generation evaluation used the v2 test split without overwriting
earlier evaluation artifacts:

- Dataset: 1,472 test examples from 42 puzzles.
- JSON parse rate: `100.00%`.
- Schema-valid rate: `100.00%`.
- Correct task-label rate: `100.00%`.
- Exact answer rate: `95.52%` (`1,406/1,472`).
- Equal-weight task macro exact rate: `96.97%`.
- Mean generation time: `1.52` seconds per example.
- Candidate-location exact rate: `82.34%`; empty cases `92.39%`, non-empty
  cases `72.28%`.
- Naked-single scan exact rate: `99.46%`.
- Coordinate, cell, unit, and peer tasks: `100.00%` exact individually.

Comparison with the earlier approximately 60%-of-epoch checkpoint evaluation
(`artifacts/results/validation/representation/sudoku-representation-v2-scratch-60pct.test.metrics.json`):

| Metric | 60% checkpoint | Full epoch | Change |
| --- | ---: | ---: | ---: |
| Overall exact answer | 91.98% (1,354/1,472) | 95.52% (1,406/1,472) | +3.53 pp |
| Task-macro exact answer | 94.66% | 96.97% | +2.31 pp |
| Candidate-location exact | 67.93% | 82.34% | +14.40 pp |
| Candidate empty-set exact | 84.24% | 92.39% | +8.15 pp |
| Candidate non-empty exact | 51.63% | 72.28% | +20.65 pp |
| Non-empty mean recall | 83.56% | 91.65% | +8.09 pp |
| Non-empty mean precision | 77.56% | 90.44% | +12.88 pp |
| Non-empty micro recall | 83.80% | 91.53% | +7.72 pp |
| Non-empty micro precision | 75.94% | 89.34% | +13.40 pp |
| Naked-single scan exact | 100.00% | 99.46% | -0.54 pp |

The biggest gain is candidate-location reasoning on non-empty answers: both
recall and precision improved substantially. The small naked-single decrease is
within the expected noise of 184 test examples. Both evaluations retained
100.00% parse, schema, and task-label rates. The 60% and full-epoch evaluations
used the same dataset SHA-256, model revision, deterministic decoding, and
test split; their evaluation batch sizes were 8 and 4 respectively.

Per-task list precision and recall comparison:

| Task | 60% exact | Full exact | Exact change | 60% mean P / R | Full mean P / R | Mean P / R change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate locations | 67.93% | 82.34% | +14.40 pp | 77.56% / 83.56% | 90.44% / 91.65% | +12.88 / +8.09 pp |
| Naked-single scan | 100.00% | 99.46% | -0.54 pp | 100.00% / 100.00% | 99.43% / 100.00% | -0.57 / +0.00 pp |

For candidate locations, micro precision improved from `75.94%` to `89.34%`
(`+13.40 pp`) and micro recall improved from `83.80%` to `91.53%`
(`+7.72 pp`). For naked-single scans, micro recall stayed at `100.00%` while
micro precision decreased from `100.00%` to `99.36%` (`-0.64 pp`). The other
tasks do not have list precision/recall metrics in this evaluator; their exact
accuracy remained `100.00%` for both checkpoints.

The full-epoch candidate-location error profile is 65 failures out of 368
test queries: 39 box queries, 19 column queries, and 7 row queries. By result
size, 31 failures are `few` (2-3 cells), 17 are `many` (4+ cells), 14 are
empty, and 3 contain one cell. The failures include 33 extra-cell-only cases,
20 missing-cell-only cases, and 12 cases with both. This indicates that the
remaining gap is concentrated in unit scanning and exact set enumeration,
especially boxes and multi-cell non-empty results, rather than JSON formatting.

### Recommended Next Representation Experiment

Do not spend another generic epoch on the full mixed dataset. Build a
candidate-location augmentation from training states only, preserving the
existing puzzle-level validation and test split. Sample additional queries
with explicit balance across row, column, and box units and across empty, one,
few, and many result sizes, with extra weight on non-empty box queries and
multi-cell outputs. The v2 builder currently supplies only two candidate
queries per state, one empty and one non-empty.

Continue from the full-epoch adapter in a new output directory with a low
learning rate and a short targeted run. Include a replay fraction of the
original v2 tasks and validate all six tasks, so candidate gains do not trade
away the current 100% performance on cell, coordinate, unit, or peer tasks.
Use the validation split for iteration and reserve the frozen test split for
the final comparison.

An exploratory 12-query-per-state augmentation was built first. It is larger
than needed for a light continuation and is retained only as an alternative:

- Builder: `evaluation/build_candidate_location_augmentation.py`.
- Artifact: `artifacts/datasets/training/representation/sudoku-candidate-location-augmentation-v1.jsonl`.
- Report: `artifacts/results/validation/representation/sudoku-candidate-location-augmentation-v1.report.json`.
- Rows: 19,032 train and 3,024 validation.
- States: 1,586 train and 252 validation.
- Queries per state: 12.
- Test states excluded: 184.
- All 12 unit/result-size categories are represented; non-empty box categories
  receive twice the base sampling weight.

Build command:

```powershell
python evaluation\build_candidate_location_augmentation.py --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output artifacts\datasets\training\representation\sudoku-candidate-location-augmentation-v1.jsonl --report artifacts\results\validation\representation\sudoku-candidate-location-augmentation-v1.report.json --queries-per-state 12 --seed 42
```

The augmentation is intentionally separate from the frozen v2 dataset. For an
apples-to-apples light continuation, use a train-only 4-query-per-state
artifact below. It adds 6,344 train rows, making the combined input 19,032
train rows while the original v2 validation set remains unchanged at 2,016
rows. The original v2 tasks still provide replay and the historical validation
comparison remains valid:

- Artifact: `artifacts/datasets/training/representation/sudoku-candidate-location-augmentation-lite-v1.train.jsonl`.
- Report: `artifacts/results/validation/representation/sudoku-candidate-location-augmentation-lite-v1.train.report.json`.
- Queries per state: 4.
- Test states excluded: 184.

Build command:

```powershell
python evaluation\build_candidate_location_augmentation.py --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output artifacts\datasets\training\representation\sudoku-candidate-location-augmentation-lite-v1.train.jsonl --report artifacts\results\validation\representation\sudoku-candidate-location-augmentation-lite-v1.train.report.json --queries-per-state 4 --seed 42 --split train
```

The two-split light artifact remains available for targeted validation
diagnostics, but it should not replace the original v2 validation protocol.

Train the light augmentation from the full-epoch adapter in a new directory:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --input artifacts\datasets\training\representation\sudoku-candidate-location-augmentation-lite-v1.train.jsonl --initial-adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1 --batch-size 2 --gradient-accumulation 4 --learning-rate 0.00005 --max-steps 400 --save-steps 10 --save-total-limit 3
```

### Candidate-Lite Result

The light continuation completed at checkpoint 400 from the full-epoch v2
adapter. It used the unchanged v2 validation set and the frozen v2 test
protocol.

- Training loss: `0.00983`.
- Validation loss: `0.00108`.
- Validation teacher-forced token accuracy: `99.97%`.
- Overall exact test accuracy: `98.44%` (`1,449/1,472`), up from `95.52%`.
- Task-macro exact accuracy: `98.91%`, up from `96.97%`.
- Candidate-location exact accuracy: `94.02%`, up from `82.34%`.
- Candidate non-empty mean precision/recall: `98.62%` / `98.19%`, up from
  `90.44%` / `91.65%`.
- Candidate non-empty micro precision/recall: `98.11%` / `97.74%`, up from
  `89.34%` / `91.53%`.
- Candidate box exact accuracy: `93.22%`, up from `66.95%`.
- Parse, schema, and task-label rates: `100.00%`.
- Other task exact accuracy remained `100.00%`, except naked-single scan at
  `99.46%`.

Evaluation artifacts:
`artifacts/results/validation/representation/sudoku-representation-v2.candidate-lite.test.metrics.json` and
`artifacts/results/validation/representation/sudoku-representation-v2.candidate-lite.test.predictions.jsonl`.

Evaluation artifacts:
`artifacts/results/validation/representation/sudoku-representation-v2.scratch-20pct.test.metrics.json` and
`artifacts/results/validation/representation/sudoku-representation-v2.scratch-20pct.test.predictions.jsonl`.

### Deduction Prompt-Format V2 Preparation

The representation adapter was trained on a labeled, box-separated grid. The
original deduction dataset used plain nine-row prompts, so separate v2 prompt
copies were created without changing the old artifacts:

- Training/validation/test dataset:
  `artifacts/datasets/training/deduction/sudoku-deductions-v2.jsonl`.
- Matching held-out benchmark:
  `artifacts/datasets/validation/deduction/sudoku-deduction-benchmark-v2.jsonl`.
- Prompt format: `sudoku-representation-v2` with `c1`-`c9` labels and explicit
  3x3 box separators.
- Deduction dataset rows: 1,678 train, 268 validation, 197 test.
- Benchmark rows: 1,094 across 401 puzzles.
- All IDs, targets, splits, puzzle counts, and logical validation results match
  the v1 copies. The v1 files remain unchanged.

The new files pass `evaluation/validate_sudoku_deduction_dataset.py`. The
converter is `evaluation/format_sudoku_deduction_prompts.py`.

Warm-start smoke command:

```powershell
python train_sudoku.py smoke --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\deduction\sudoku-deductions-v2.jsonl --initial-adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1 --batch-size 2 --gradient-accumulation 4
```

The planned full warm-start run should use a new output directory and the v2
deduction dataset. Its benchmark evaluation must use the matching v2 benchmark
copy, not the old prompt-format benchmark.

Recorded continuation command:

```powershell
python train_sudoku_representation.py run --model-id Qwen/Qwen3.5-2B --input artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --output-dir artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --batch-size 2 --gradient-accumulation 4 --learning-rate 0.0002 --max-steps 1586 --save-steps 10 --save-total-limit 3 --resume
```

Recorded evaluation command:

```powershell
python evaluation\evaluate_sudoku_representation.py --dataset artifacts\datasets\training\representation\sudoku-representation-v2.jsonl --adapter artifacts\checkpoints\representation\sudoku-representation-qwen3.5-2b-qlora-v2-scratch-20pct --split test --predictions artifacts\results\validation\representation\sudoku-representation-v2.scratch-20pct.test.predictions.jsonl --metrics artifacts\results\validation\representation\sudoku-representation-v2.scratch-20pct.test.metrics.json --batch-size 4 --overwrite
```

### Dataset-overlap audit

On 2026-09-18, `evaluation/analyze_dataset_overlap.py` checked deduction and
representation training data against both the deduction benchmark and the
1,000-puzzle final benchmark. Exact puzzle, digit-renamed puzzle, complete
state, normalized prompt, and solution-hash overlap were all zero. The 500
deduction benchmark source traces also had zero overlap with all 452 training
puzzles. Internal train/validation/test puzzle overlap was zero for both
datasets.

The earlier resume error was an operational path mismatch: it targeted the v1
representation run while the latest checkpoint belonged to the v2 scratch run.
The v1 run also had a different dataset fingerprint, so the resume guard
correctly rejected it. No checkpoint was corrupted.

### 2026-09-20: Representation warm-start deduction run

The latest representation adapter was continued on the v2 deduction dataset,
whose prompts use the labeled, box-separated grid format:

- Adapter: `artifacts/checkpoints/deduction/sudoku-deduction-qwen3.5-2b-qlora-warm-v2/`.
- Initial adapter: `artifacts/checkpoints/representation/sudoku-representation-qwen3.5-2b-qlora-candidate-lite-v1/`.
- Dataset: `artifacts/datasets/training/deduction/sudoku-deductions-v2.jsonl`.
- Configuration: 3 epochs, batch size 2, gradient accumulation 4, effective batch size 8, learning rate `0.0002`.
- Training loss: `0.24172`; validation loss: `0.20112`; validation token accuracy: `92.37%`.
- Benchmark: `artifacts/datasets/validation/deduction/sudoku-deduction-benchmark-v2.jsonl`, 1,094 examples.
- Parse rate: `100.00%`.
- Sound/valid deduction rate: `0.914%` (`10/1,094`).
- Exact target rate: `0.00%`.
- Status: completed and evaluated.

The warm-start doubled the valid-deduction rate of the earlier plain-prompt
Qwen deduction adapter (`0.457%`, 5/1,094), while improving parse reliability
from the representation-only adapter. It still does not reliably perform
deduction: all ten valid outputs came from simple or subset techniques, with no
valid pointing, claiming, or fish deductions. The adapter therefore improved
format adherence and produced a small reasoning signal, but is not yet usable
as a Sudoku deduction engine.
