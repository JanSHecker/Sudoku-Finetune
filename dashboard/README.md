# Evaluation Dashboard

The dashboard indexes metric and prediction artifacts under `artifacts/` and
derives imbalance-aware metrics from the prediction records. It is intentionally
local: no credentials or raw artifacts leave the machine.

## Run

From the repository root:

```powershell
python -m pip install -r dashboard\requirements.txt
streamlit run dashboard\app.py
```

Use the sidebar to change the artifact directory, rescan files, switch views,
and focus on a run. The loader recognizes the current deduction,
representation, baseline, and QLoRA training schemas. Future evaluators should
include a `predictions` path in their metrics JSON; the loader also supports the
existing `*.metrics.json` / `*.predictions.jsonl` naming convention.

## What To Trust

Macro precision, macro recall, macro F1, and the confusion matrix give every
observed class equal influence. Weighted and micro metrics remain available for
operational volume, while raw accuracy is not used as the sole headline score.

For Sudoku deductions, logical soundness is shown separately from predicted
rule-family classification. A valid deduction can use a different rule from
the selected target, so those metrics answer different questions.
