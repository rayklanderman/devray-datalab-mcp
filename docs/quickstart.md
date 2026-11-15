# Quickstart: End-to-End ML Workflow

This tutorial walks through a complete machine learning workflow using DevRay DataLab MCP
with the included example dataset `examples/sample_classification.csv`.

All processing runs **locally** on your computer. No cloud, no API keys, no paid services.

## Prerequisites

- DevRay DataLab MCP installed: `pip install devray-datalab-mcp`
- Connected to an MCP-compatible client (Kilo Code, Cline, Cursor, Claude Desktop, VS Code)
- The example dataset available at `examples/sample_classification.csv` (copy it into your
  configured `DEVRAY_WORKSPACE_DIR`)

## Step 1: Configure your local dataset directory

Set `DEVRAY_WORKSPACE_DIR` to a folder on your machine, then copy the example dataset into it:

```powershell
$env:DEVRAY_WORKSPACE_DIR="C:\MLData\datasets"
$env:DEVRAY_ARTIFACTS_DIR="C:\MLData\artifacts"
Copy-Item "examples/sample_classification.csv" "C:\MLData\datasets\"
```

## Step 2: Inspect the dataset

Ask your AI assistant:

> "Use inspect_dataset to load sample_classification.csv and tell me about the columns, data types, missing values, and row count."

Expected result includes: 20 rows, 6 columns (age, income, credit_score, category, region, purchased), no missing values, no duplicate rows, target `purchased` is binary (0/1).

## Step 3: Check data quality

> "Run analyze_data_quality on sample_classification.csv and report any concerns."

Expected: no missing values, no duplicates, no constant columns, no severe outliers — data is clean and ready for modeling.

## Step 4: Analyze correlations

> "Use analyze_correlations with Pearson method on sample_classification.csv and show the strongest positive and negative pairs."

Expected: identifies correlations among age, income, credit_score, and purchased (e.g., income and credit_score likely correlate positively with purchased).

## Step 5: Train and compare models

> "Use compare_models to train models predicting 'purchased' from sample_classification.csv using classification task, comparing logistic_regression, decision_tree, and random_forest, selecting by F1."

Expected: trains all three models on consistent train/test splits, reports accuracy/precision/recall/F1/ROC-AUC, and selects the best model by F1.

## Step 6: Export the best model

> "Use export_model to save the trained random_forest model as 'churn_model' predicting 'purchased' from sample_classification.csv."

Expected: saves a joblib artifact containing the model, preprocessing pipeline, feature schema, hyperparameters, metrics, and package versions under `DEVRAY_ARTIFACTS_DIR`.

## Step 7: Run predictions

> "Use predict with artifact_name 'churn_model' and features: age 40, income 80000, credit_score 720, category electronics, region east."

Expected: returns a prediction (0 or 1) plus class probabilities.

## Complete prompt sequence (copy-ready)

```
inspect_dataset: {"name": "sample_classification.csv"}
analyze_data_quality: {"name": "sample_classification.csv"}
analyze_correlations: {"name": "sample_classification.csv", "method": "pearson"}
compare_models: {"name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "models": ["logistic_regression", "decision_tree", "random_forest"], "metric": "f1"}
export_model: {"artifact_name": "churn_model", "name": "sample_classification.csv", "target_column": "purchased", "task_type": "classification", "model_name": "random_forest", "random_state": 42}
predict: {"artifact_name": "churn_model", "features": {"age": 40, "income": 80000, "credit_score": 720, "category": "electronics", "region": "east"}}
```

## Regression example

Repeat the same workflow with `examples/sample_regression.csv`, target `spend`, task type `regression`, and selection metric `r2`.

## Key behaviors to know

- **Preprocessing is fit on training data only** — no data leakage.
- **Train/test split** is applied before fitting the preprocessing pipeline.
- **Stratified splitting** is used for classification where valid.
- **Predictions validate the feature schema** — missing required features produce a clear `schema_mismatch` error.
- **Model artifacts are local only** — stored under `DEVRAY_ARTIFACTS_DIR`.
