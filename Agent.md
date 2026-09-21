# Project Agent Instructions

## Experiment History

Before changing training, evaluation, dataset-generation, or experiment
configuration, always inspect the current experiment history first:

- Read `README.md` and the relevant sections of `evaluation/README.md`.
- Inspect matching local artifacts, especially `run_config.json`,
  `run_status.json`, `metrics.json`, checkpoint `trainer_state.json`, and dataset
  reports.
- Treat artifact metadata as the source of truth for completed runs, including
  the dataset fingerprint, model revision, checkpoint, step count, and metrics.
- Check whether a run is completed, interrupted, or failed before selecting a
  checkpoint or proposing a resume command.

After every meaningful experiment or resume:

- Update the experiment history in `README.md` with the date, dataset, output
  directory, checkpoint transition, effective batch size, step target, result,
  and status.
- Record the exact command or enough command arguments to reproduce it.
- Record failed or rejected resume attempts when they affect reproducibility,
  and explain the correction without deleting prior history.
- Record dataset-overlap checks before treating benchmark results as valid.
- Keep durable summaries in tracked documentation because `artifacts/` is
  ignored by Git.

Never overwrite an earlier experiment entry. Add a new dated entry when a run
continues, changes datasets, changes prompts, or changes training settings.
