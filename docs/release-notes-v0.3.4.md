# DevRay DataLab MCP v0.3.4 release notes draft

DevRay DataLab MCP v0.3.4 adds model selection to leaderboard calls and improves the guidance
available to assistants through all 27 MCP tool descriptions. The server runs over stdio on
Python 3.11 or later and requires no cloud account or API key.

## Compare specific models

Call `get_model_leaderboard` with an existing workspace dataset, its target column, and the task
type. The optional `models` array accepts unique model IDs for that task. For example, with the
classification sample from this repository:

```json
{
  "name": "sample_classification.csv",
  "target_column": "purchased",
  "task_type": "classification",
  "models": ["decision_tree", "logistic_regression"],
  "metric": "f1",
  "max_trials": 2
}
```

Each call runs fresh cross-validation and returns the ranked trials, the winner and its score,
and whether Optuna ran. Omit `models` to use the previous default candidates. Ties follow evaluation
order, and the trial and time budgets can limit which candidates are scored. Invalid selections
return `validation_error`. Use `automl_run` for full run metadata.

The AutoML tools now honor `delimiter` for CSV files. Optuna is optional and runs only when
installed and tuning budget remains. Pass a `split_manifest_id` to reserve the manifest's test
rows for evaluation.

## Reliability improvements

- Timestamp split manifests preserve chronological order for CV, including on reuse of legacy manifests.
- Candidate evaluation and Optuna share a monotonic deadline. A trial in progress can finish after
  the deadline.
- Seeded tuning is reproducible, and the winner comes from the highest-ranked successful trial.
- Cooperative training cancellation records the cancelled terminal state.
- Explicit tuning accepts every supported model ID for the given task.
- Prediction returns `validation_error` for invalid artifact identifiers.

## Installation and Glama

Use Python 3.11+ and launch the installed `devray-datalab-mcp` command from a stdio client. Set
`DEVRAY_WORKSPACE_DIR` to the server-side dataset directory and `DEVRAY_ARTIFACTS_DIR` to a writable
directory. Set `DEVRAY_EXPERIMENTS_DB_PATH` to a file under a writable directory when the client
launches the process from a read-only working directory.

Glama can build from PyPI or infer a container build and wrap the existing stdio command. Datasets
must be readable in the environment running the server. Hosted verification for this release
candidate is pending. [Glama hosting documentation](https://glama.ai/mcp/hosting)
