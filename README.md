# Distributed fault prediction for cyber-physical systems on NASA C-MAPSS

Notebooks, configuration and saved results for a study of a **distributed, lightweight fault-prediction framework**: five simulated nodes each train a small model on their own engines, and their predictions are combined, instead of one central model seeing everything. The comparison point is a centralised baseline, and the focus is robustness (noise, missing values, sensor failure, node dropout).

**Authors:** Aaditya Upadhyay, Dr. Akshay Jadhav (guide)
**Context:** Project Based Learning 4, VI semester (Jan to May 2026), Computer Science and Engineering, Manipal University Jaipur
**Status:** manuscript in preparation, targeting IEEE ICPS 2026

> This is not the EEG stress paper. That work lives in [EEG-Stress-Detection-](https://github.com/Addy48/EEG-Stress-Detection-).

## Dataset

NASA C-MAPSS **FD001**: 100 turbofan engines run to failure, 14 informative sensors after dropping 7 constant ones, 20,631 cycles. A label is positive when a cycle is within W cycles of the engine's last cycle (W in 10, 20, 30). Engines are split 70 / 15 / 15 into train, validation and test, and the 70 training engines are spread over five nodes of 14. FD003 is used once, as a cross-dataset check. The raw files come from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and are not stored here. See [`data/README.md`](data/README.md).

## Results

Held-out test engines (15), from `results/tables/`:

| Metric | Centralised | Distributed |
|---|---:|---:|
| Precision | 0.871 | 0.799 |
| Recall | 0.813 | 0.883 |
| **F1** | **0.841** | **0.839** |

The two systems are statistically indistinguishable on the test engines (Wilcoxon p = 0.524, `wilcoxon_test_results.csv`). The best centralised model on validation (Random Forest, window 20) reaches F1 0.864.

Where they differ is under degradation. Mean F1 across each experiment's levels (`robustness_summary.csv`):

| Experiment | Centralised | Distributed |
|---|---:|---:|
| Sensor noise | 0.841 | 0.853 |
| Missing values | 0.831 | 0.856 |
| Sensor failure | 0.794 | 0.854 |
| Node dropout | 0.190 | 0.859 |

With 30% of nodes dropped, the centralised F1 is 0.000 and the distributed one is 0.856. Mean detection delay on the 15 test engines is -3.2 cycles (median -3; `detection_delay_results.csv`). On FD003, a different dataset, the distributed framework reaches F1 0.753 (precision 0.874, recall 0.661).

Read these with the scale in mind: one benchmark family, 15 test engines, simulated nodes on one machine.

## Pipeline

Run the notebooks in order. Each one lists what it reads and writes in its first cell.

| # | Notebook | Output |
|---|---|---|
| 01 | [`01_data_preprocessing`](notebooks/01_data_preprocessing.ipynb) | cleaned dataset |
| 02 | [`02_window_labeling_partition`](notebooks/02_window_labeling_partition.ipynb) | labelled dataset, engine split |
| 03 | [`03_feature_engineering`](notebooks/03_feature_engineering.ipynb) | rolling features, scaler |
| 04 | [`04_centralized_baseline_models`](notebooks/04_centralized_baseline_models.ipynb) | baseline comparison, best model |
| 05 | [`05_node_simulation`](notebooks/05_node_simulation.ipynb) | per-node models, F1 weights |
| 06 | [`06_aggregation_strategies`](notebooks/06_aggregation_strategies.ipynb) | majority, F1-weighted and risk-based aggregation |
| 07 | [`07_robustness_experiments`](notebooks/07_robustness_experiments.ipynb) | robustness tables |
| 08 | [`08_statistical_analysis`](notebooks/08_statistical_analysis.ipynb) | comparison and summary tables |
| 09 | [`09_final_visualizations`](notebooks/09_final_visualizations.ipynb) | figures |
| 10 | [`10_paper_experiments`](notebooks/10_paper_experiments.ipynb) | test comparison, Wilcoxon, detection delay, FD003 |

The notebooks were written for Kaggle: paths point to `/kaggle/input/datasets/<kaggle-user>/...` and outputs go to `/kaggle/working/`. To run them elsewhere, change `BASE` and `OUT` in the first code cell.

## Repository layout

```
notebooks/            the ten notebooks, in run order (outputs kept)
config/config.json    engine split, node layout, dropped sensors, window sizes
data/intermediate/    cleaned, labelled and feature datasets produced by notebooks 01 to 03
data/README.md        where the raw data comes from and how the files connect
results/tables/       result CSVs behind every number above
results/figures/      the paper figures
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
```

## Known gaps

- The original `config.json` lived in a Kaggle dataset and was not saved. `config/config.json` was rebuilt from the saved outputs and data (see its `_note`); the validation and test engine IDs (71 to 85 and 86 to 100) are confirmed for test and inferred for validation.
- Trained models (`.pkl`) are not included, only the datasets and result tables.
- No licence has been chosen yet.
