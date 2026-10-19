# Results

Saved outputs of the notebooks; nothing here was edited by hand. [`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md) says what each can support.

| Folder | Contents |
|---|---|
| `by_notebook/nb03_centralized/` | 120 runs (6 models x 4 datasets x 5 seeds): `centralized_results.csv`, `threshold_summary.csv`, `deployment_metrics.csv` |
| `by_notebook/nb04_distributed/` | 120 runs (2 partitions x 3 aggregations x 4 datasets x 5 seeds) |
| `by_notebook/nb05_robustness/` | 1,440 runs (4 perturbations x 3 levels x 6 models x 4 datasets x 5 seeds); summary |
| `by_notebook/nb06_cross_dataset/` | 360 runs (12 train-test pairs x 6 models x 5 seeds); generalisation summary |
| `by_notebook/nb07_rul/` | 60 runs (3 regressors x 4 datasets x 5 seeds) |
| `master/` | merged tables from notebook 08. They mix ML and DL rows (for example the node-dropout rows are all DL) and contain repeated deployment rows, so filter by `Source` or `model` before counting |
| `paper_tables/` | Table 1 to 9 as CSV and PNG |
| `figures/` | the paper figures as PNG, PDF and SVG |
| `statistical_tests/` | Friedman and Wilcoxon output, average ranks |
| `dl_baselines/` | result files of the separate DL pipeline (LSTM, GRU, TCN, Transformer): centralised, distributed, robustness, cross-dataset and deployment. Seeds 42, 123, 256, 512, 1024 |
| `SUMMARY.md` | tables rebuilt from `by_notebook/` and `dl_baselines/` by `scripts/build_summary.py` |
