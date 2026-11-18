# Glama baseline before v0.3.4 changes

Observed November 18, 2026; local source baseline `7d2305e2d878ae462a92de22e6c9bf26b277bf9d`. Sanitized equivalent: `193279e0e718a294d431c2ea984933c9630c73ff`.
Sources: [Glama Overview](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp),
[Schema](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp/schema),
[Source](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp/tree),
[Admin](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp/admin),
and [Glama Hosting](https://glama.ai/mcp/hosting).

## Discovery, indexing, and runtime

- The public Python/Local listing is indexed and shows 27 tools. The Overview's November 17
  assessment is **B (3.1/5)**: disambiguation 3/5, naming 4/5, count 3/5, completeness 4/5.
  The Schema page shows an older November 15 ten-tool assessment (**B, 3.2/5**), but lists
  27 tools. All 27 individual pages were read; five detail grades differ from the Overview.
  The most consistent findings were weak usage guidance (2–3/5 on every tool), incomplete
  side-effect descriptions, and little explanation of how input parameters interact. The table
  combines those findings with implementation checks; grader suggestions are not runtime facts.
- The Schema snapshot advertises MCP `2025-11-25` with tools and no resources or prompts.
  This establishes discovery, not a successful tool call, active instance, or correct deployment.
  The Deploy button is not evidence of a running server. The admin page requires maintainer
  sign-in for build details and re-sync. No hosted build logs or tool calls were verified and
  no Glama runtime or stdio failure was confirmed.
- No Dockerfile or `.dockerignore` existed in the public source tree or accepted checkout.
  The PyPI console script starts a local stdio server, logging diagnostics to stderr. Glama
  supports PyPI and Docker builds and can wrap stdio; a Dockerfile provides an explicit build
  path, not a prerequisite for every Glama deployment. Set writable artifact/SQLite paths,
  allow read access to fixture data, and do not bundle user data, credentials, or model artifacts.
  No ownership `glama.json` is required for claiming a personal GitHub repository.
- **Confirmed interface defect:** the leaderboard starts a new search and neither advertises
  nor forwards `models`; `automl_run` does. AutoML core validation also used truthiness for
  explicit subsets, so `[]` meant defaults and malformed values could fail inconsistently.
  All three AutoML schemas advertised a CSV delimiter that their loaders did not receive.
  These are source-confirmed issues, not evidence of a failed Glama deployment.
- The [PyPI listing](https://pypi.org/project/devray-datalab-mcp/) showed `0.3.2`; a direct
  [Registry API](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.rayklanderman%2Fdevray-datalab-mcp)
  fetch confirmed `0.3.2` as latest (an older cached response still reported `0.3.1`).
  The checkout's `server.json` described PyPI `0.3.2` over stdio.

## All 27 tool assessments

| Tool | Overview | Detail (/5) | Actionable finding |
|---|:---:|:---:|---|
| `list_datasets` | B | B 3.4 | Say where it looks, which formats it lists, and which tool profiles a file. |
| `inspect_dataset` | C | B 3.4 | Specify preview bounds, optional format extras, and raw sample exposure. |
| `analyze_correlations` | C | C 2.9 | Describe numeric-only analysis, Pearson vs. Spearman, and avoid causal claims. |
| `analyze_data_quality` | C | C 2.6 | List checks and clarify that it recommends fixes but does not alter data. |
| `analyze_feature_drift` | B | C 2.9 | Identify baseline vs. current inputs, common columns, and alpha semantics. |
| `detect_potential_leakage` | C | C 2.6 | Identify the label column as optional and results as heuristic, not proof. |
| `prepare_dataset` | C | C 2.7 | Explain train-only preprocessing and that its artifact is not a predictive model. |
| `train_baseline` | C | B 3.2 | Distinguish a synchronous single-model evaluation from jobs, export, and comparison. |
| `compare_models` | C | C 2.8 | Explain new CV fits vs. history lookup, metric semantics, and holdout usage. |
| `automl_run` | C | C 2.9 | Explain fresh search, candidate order, metric/budget, and optional Optuna. |
| `tune_model` | C | B 3.1 | Specify one model, CV only, optional Optuna, and no exported fitted model. |
| `get_model_leaderboard` | C | C 2.9 | Say it reruns search, not a lookup; align subset, metric, and budget semantics. |
| `start_training_job` | B | B 3.2 | State single-model scope, asynchronous behavior, and how to poll. |
| `get_training_status` | B | B 3.1 | Identify the job ID source; distinguish terminal results from experiment records. |
| `cancel_training_job` | B | B 3.4 | Explain cancellation is cooperative and status must be polled. |
| `list_training_jobs` | B | B 3.1 | Explain process-local job history vs. persisted experiments. |
| `export_model` | C | C 2.6 | Distinguish saved-handle export from train-and-export and describe writes. |
| `predict` | C | C 2.9 | Specify keyed feature rows, required schema, and trusted local artifacts. |
| `list_experiments` | A | A 3.5 | Minor clarification: persisted run history, not jobs or exported artifacts. |
| `get_experiment` | B | B 3.2 | Identify ID source and output; does not train or load a model. |
| `compare_experiments` | C | C 2.9 | Contrast saved metrics with new fits; only compare comparable runs. |
| `list_models` | C | C 2.8 | Specify tracking records, not a filesystem inventory or availability check. |
| `load_model_handle` | C | C 2.9 | Clarify input alternatives, metadata output, and artifact trust. |
| `create_split_manifest` | B | C 2.9 | Clarify split-column choices, fractions, persistence, and holdout. |
| `validate_split_manifest` | C | C 2.9 | Explain report semantics, dataset match, and no repair/rewrite. |
| `list_split_manifests` | A | A 3.7 | Minor clarification: lists stored metadata but does not validate datasets. |
| `system_info` | A | A 4.0 | Minor clarification: hardware visibility is not GPU acceleration support. |

Glama also noted naming variation, overlapping training/listing surfaces, and lack of cleanup
operations. Keep the stable 27 names and capabilities; distinguish choices in descriptions
rather than consolidating tools or adding mutation APIs. A grade is a discoverability signal,
not a runtime test or a guarantee that any particular parameter is supported.

## Continuation check

- The Overview also served a November 17 assessment of **B (3.2/5)** across 27 tools. The public
  leaderboard detail remained **C (2.9/5)** and its captured input schema omitted `models`.
  The published schema history records successful inspections of the existing server.
  [Glama Overview](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp)
  [Leaderboard assessment](https://glama.ai/mcp/servers/rayklanderman/devray-datalab-mcp/tools/get_model_leaderboard)
- Glama can infer builds for indexing and supports deployment from PyPI or a Dockerfile. A
  repository Dockerfile could standardize the image and file permissions when needed. The
  indexed schema and supported PyPI/stdio path were sufficient to keep Docker packaging out of
  this candidate. A build or tool call for v0.3.4 in Glama's environment still needs verification.
  [Indexing methodology](https://glama.ai/mcp/methodology)
  [Hosting documentation](https://glama.ai/mcp/hosting)
