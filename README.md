# Distributed fault prediction for cyber-physical systems on NASA C-MAPSS

Notebooks, configuration and saved results for a classical machine-learning study of fault prediction on all four NASA C-MAPSS turbofan datasets (FD001 to FD004). It compares **centralised** models with a **five-node distributed** setup, and measures robustness, cross-dataset transfer, remaining-useful-life regression and deployment cost.

**Authors:** Aaditya Upadhyay, Dr. Akshay Jadhav (guide)
**Context:** Project Based Learning 4, Computer Science and Engineering, Manipal University Jaipur
**Status:** manuscript in preparation. Not published.

> Not the EEG stress paper: that is [EEG-Stress-Detection-](https://github.com/Addy48/EEG-Stress-Detection-). An earlier single-dataset (FD001) version of this repository is kept at the git tag [`try2-single-dataset`](https://github.com/Addy48/cps-fault-prediction/tree/try2-single-dataset); its numbers are superseded by this pipeline.

## Setup of the study

| | |
|---|---|
| Data | C-MAPSS FD001 to FD004; 15 active signals after dropping 6 constant sensors; 18 base features (15 sensors + 3 operating settings) |
| Label | failure within 20 cycles (RUL <= 20); positive rate 8.5 to 10.2% |
| Split | by engine, never by row: train / validation A / validation B / test (for example FD001: 70 / 9 / 5 / 16 engines). Scaler fitted on train only; thresholds chosen on validation B |
| Features | 342 = rolling mean, std, min, max, slope and EMA over windows 5, 10, 20 |
| Models | logistic regression, SVM, random forest, XGBoost, LightGBM, CatBoost |
| Distributed | five nodes, each with its own engines and a LightGBM model; partition by engine-ID block or by k-means on operating signature; majority, F1-weighted and confidence voting |
| Seeds | 42, 52, 62, 72, 82 (see the note on seeds below) |

## What the saved results show

Everything below is rebuilt from the result files by [`scripts/build_summary.py`](scripts/build_summary.py); the full tables are in [`results/SUMMARY.md`](results/SUMMARY.md).

**Centralised classification (F1).** The best single result is LightGBM on FD003 at 0.947 (AUROC 0.999). Averaged over the four datasets the top three are a tie: logistic regression 0.878, LightGBM 0.875, SVM 0.874. FD004, the hardest set, tops out at 0.815. Gradient boosting is not clearly better than a linear model here.

**Distributed versus centralised.** With LightGBM at every node, the best distributed setting (engine-ID block partition with confidence voting) reaches F1 0.885, 0.820, 0.920 and 0.786 on FD001 to FD004. That is 0.013 to 0.032 *below* centralised LightGBM. Engine-ID block beats operational clustering on all four datasets under confidence voting.

**Cross-dataset transfer.** Models trained on FD002, FD003 or FD004 score about 0.90 F1 on FD001. Transfer to FD003 gives 0.48 to 0.51, and anything to FD002 or FD004 gives 0.16 to 0.20.

**RUL regression.** Best RMSE on FD001 is 34.5 cycles (LightGBM). On FD002 to FD004 random forest is best (46.9, 56.8, 73.9).

**Deployment.** Median single-sample CPU latency is 0.005 to 0.04 ms for the six ML models, with sizes from 0.02 MB (logistic regression) to 20 MB (random forest, in the classification runs). The DL baselines measured 1 to 9 ms.

**ML versus DL baselines.** The DL baselines (LSTM, GRU, TCN, Transformer) were produced by a separate pipeline and are included as result files only. Mean F1 over the four datasets: GRU 0.890, Transformer 0.881, best ML (logistic regression) 0.878.

## Read this before using a number

The robustness and statistics results have problems that change what they can support. The details are in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). In short:

- **ML robustness was run on centralised models only.** There is no ML node-dropout experiment and no ML replicated-centralised baseline, so the study does not yet test the claim that distribution makes the ML models more robust. The node-dropout numbers in `results/master/` belong to the DL baselines.
- **The ML perturbations collapse the models** (clean F1 0.867; Gaussian noise about 0.13 at every level) because they are applied to the 342-column engineered matrix. The DL baselines kept F1 0.69 to 0.81 under the same names. The two are not comparable and the ML noise result is probably an implementation artefact.
- **The "five seeds" are not five independent runs.** The split is fixed, so logistic regression, SVM, XGBoost and LightGBM return the same F1 every seed (std 0). Only random forest and CatBoost vary.
- **The statistical tests have four datasets as their units.** The smallest possible Wilcoxon p-value is 0.125, and the rank table includes a meaningless "Other" entry.
- **Notebook 03 (centralised baselines) is missing.** Its outputs, configuration and results are here; the notebook file was empty.

## Repository layout

```
notebooks/        01, 02, 04 to 08 (outputs kept); 03 is missing, see docs/PIPELINE.md
config/           split, sensor, feature and run configuration for every notebook
results/
  by_notebook/    result CSVs of notebooks 03 to 07 (ML pipeline)
  master/         the merged tables (ML + DL), as produced by notebook 08
  paper_tables/   Table 1 to 9 as CSV and PNG
  figures/        paper figures as PNG, PDF and SVG
  statistical_tests/
  dl_baselines/   DL result files (separate pipeline; code not included)
  SUMMARY.md      tables rebuilt by scripts/build_summary.py
docs/             PIPELINE.md, PROTOCOL.md, LIMITATIONS.md
scripts/          build_summary.py
tests/            checks that the saved files agree with this README
```

Models, predictions and feature parquet files (about 3.5 GB) are not stored here.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_summary.py     # rebuild the summary tables from the saved results
python -m pytest -q
```

To re-run the experiments, upload the C-MAPSS files to Kaggle and run notebooks 01 to 08 in order ([`docs/PIPELINE.md`](docs/PIPELINE.md)).

## Data

NASA C-MAPSS from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). Not stored here.

## Licence

None chosen yet. Until one is added, all rights are reserved.
