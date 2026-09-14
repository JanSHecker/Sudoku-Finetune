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

Artifact: `artifacts/sudoku-qwen2.5-1.5b-qlora/metrics.json`

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

The supported rule families are:

- naked and hidden singles;
- naked and hidden pairs, triples, and quads;
- pointing and claiming;
- basic fish sizes 2 through 8 in row and column orientations.

Finned, sashimi, Franken, Mutant, and Kraken fish are not implemented.

### Training dataset

Artifact: `artifacts/sudoku-deductions-v1.jsonl`

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

Artifact: `artifacts/sudoku-deduction-benchmark-v1.jsonl`

- Benchmark rows: 1,094.
- Benchmark puzzles: 401.
- 50 examples per rule in most rule categories.
- The benchmark uses separate puzzles and is not read by the trainer.
- The evaluator uses independent witness, candidate, and transition checks.

## Deduction Trainer

`train_sudoku.py` was replaced with a deduction-specific trainer. It:

- reads `artifacts/sudoku-deductions-v1.jsonl` by default;
- trains only on rows marked `train`;
- validates only on rows marked `validation`;
- excludes the `test` split from training;
- rejects malformed examples and conflicting split puzzles;
- applies completion-only loss to the JSON deduction;
- supports Qwen3.5 and LFM chat templates;
- handles LFM's mandatory `<think>` block;
- saves separate adapters for each model.

The smoke adapters are:

- `artifacts/sudoku-deduction-qwen3.5-2b-qlora-smoke`;
- `artifacts/sudoku-deduction-lfm2.5-2.6b-qlora-smoke`.

Both smoke runs completed successfully. The full Qwen adapter is:

- `artifacts/sudoku-deduction-qwen3.5-2b-qlora`.

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

Artifact: `artifacts/sudoku-deduction-qwen3.5-2b-qlora/metrics.json`

- Training epochs: 3.
- Training loss: `0.2638`.
- Validation loss: `0.2278`.
- Validation teacher-forced token accuracy: `91.10%`.
- Training plus validation runtime: approximately 49 minutes.

The low validation loss and high token accuracy show that the adapter learned
the target JSON format and many token patterns. They do not show that the model
can generate valid deductions freely.

## Held-Out Qwen Evaluation

Artifact: `artifacts/sudoku-deduction-benchmark-v1.qwen.metrics.json`

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

Artifact: `artifacts/sudoku-deduction-benchmark-v1.qwen-v2.metrics.json`

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
