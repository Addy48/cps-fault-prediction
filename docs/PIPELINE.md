# Pipeline

Eight Kaggle notebooks, each with one job. Outputs pass between them as parquet, CSV and JSON files.

| # | Notebook | Reads | Writes | In this repository |
|---|---|---|---|---|
| 01 | [`01_data_preprocessing`](../notebooks/01_data_preprocessing.ipynb) | `train/test/RUL_FD00x.txt` | per-split parquet, sensor and split metadata | notebook, `config/` |
| 02 | [`02_feature_engineering`](../notebooks/02_feature_engineering.ipynb) | notebook 01 splits | 342-column feature parquet, `feature_columns.json` | notebook, `config/` |
| 03 | centralised baselines (6 models x 4 datasets x 5 seeds) | feature parquet | `centralized_results.csv`, `threshold_summary.csv`, models | notebook not included; results and config present |
| 04 | [`04_distributed_ml_pipeline`](../notebooks/04_distributed_ml_pipeline.ipynb) | feature parquet | `distributed_results.csv`, node metadata | notebook, results |
| 05 | [`05_robustness_testing`](../notebooks/05_robustness_testing.ipynb) | test features, notebook 03 models | `robustness_results.csv` | notebook, results |
| 06 | [`06_cross_dataset_generalization`](../notebooks/06_cross_dataset_generalization.ipynb) | features, notebook 03 models | `cross_dataset_results.csv` | notebook, results |
| 07 | [`07_rul_regression`](../notebooks/07_rul_regression.ipynb) | features | `rul_results.csv` | notebook, results |
| 08 | [`08_final_analysis`](../notebooks/08_final_analysis.ipynb) | everything above plus DL baseline CSVs | master tables, Table 1 to 9, figures, statistical report | notebook, results |

## Running order and inputs

Notebooks 01 and 02 run first. Notebook 03 produces the models that 05 and 06 reload. The notebooks were written for Kaggle: input datasets are mounted under `/kaggle/input/datasets/<kaggle-user>/...` (the Kaggle account name is replaced by a placeholder) and outputs go to `/kaggle/working/`. Notebook 04 reads the features from a Kaggle dataset named `nb-3-again-outputs`, although the features come from notebook 02; that is a naming slip in the dataset, not a different pipeline.

## Configuration

`config/` holds the files the notebooks wrote for later stages: `sensor_metadata.json` (all, removed and active sensors), `split_metadata.json` (engine and row counts per split), `feature_metadata.json` and `feature_columns.json` (the 342 columns), `dataset_manifest.*`, `run_config_nb0X.json` and the notebook summaries.

## Result files

See [`results/README.md`](../results/README.md).
