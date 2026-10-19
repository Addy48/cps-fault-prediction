# Limitations and scope

What the saved results cover and how to read them. Each point was checked against the files in `results/` and the notebooks in this repository.

## 1. Which ML experiments are in the files

Notebook 05 perturbs the test features and scores the six centralised models. Notebook 04 builds the distributed system and evaluates it on clean data. Not among the files here:

- ML node-dropout results (removing nodes from the vote at inference, levels 10 to 80%);
- an ML *replicated centralised* baseline (five replicas with different seeds and the same aggregation) for the fair centralised-versus-distributed comparison the protocol describes.

The node-dropout rows in `results/master/master_robustness_results.csv` belong to the DL baselines (LSTM, GRU, TCN, Transformer). If those ML experiments exist, their notebooks and outputs should be added before quoting an ML node-dropout number. What the files here show for distribution is its clean-data cost: 0.013 to 0.032 F1 below centralised LightGBM.

## 2. How the ML perturbations were applied

Clean mean F1 over the six ML models is 0.867. Under Gaussian noise it is about 0.13 at every level (0.1, 0.2, 0.3); the other perturbations give 0.23 to 0.42. The results are reported as saved.

How they were produced (`notebooks/05_robustness_testing.ipynb`): perturbations act on the 342-column engineered feature matrix. Noise has standard deviation `level * X.std().mean()`, one scale for columns whose ranges differ widely. "Sensor failure" zeroes 10 to 30% of *feature columns*, and "corrupted sensors" replaces columns with values above their maximum. Perturbing the raw sensor readings and recomputing the features would be a different, more sensor-like test.

The DL baselines (F1 0.81, 0.74 and 0.69 under Gaussian noise at the same three levels) used their own perturbation code, so ML and DL robustness numbers are not directly comparable. Missing-value levels also differ: 10, 20, 30% for ML and 5, 10, 20% for DL.

## 3. Seeds

The split is fixed (`random_state = 42`). Seeds 42, 52, 62, 72 and 82 change only the model's internal randomness. Logistic regression, SVM, XGBoost and LightGBM are deterministic here, so all five seeds give the same F1 (std 0.000). Random forest and CatBoost vary by 0.003 to 0.029. "Mean ± std over seeds" therefore reflects model randomness only, not variation across engine splits.

The test sets are small: 16 engines (FD001, FD003) and about 40 (FD002, FD004).

## 4. Statistical tests

`results/statistical_tests/` compares models across the four datasets:

- the Friedman test (chi-square 14.33, p = 0.046) has four blocks, one per dataset;
- the pairwise Wilcoxon tests have n = 4, so the smallest attainable p-value is 0.125;
- the average-rank table includes an entry called "Other" (rank 7.0) and merges the four DL models into one "Deep Learning" entry.

Read the ranking as descriptive.

## 5. ML and DL comparisons

The DL baselines come from a separate pipeline whose code is not in this repository:

- DL seeds were 42, 123, 256, 512 and 1024;
- DL and ML latency were measured separately, and the DL FLOPs and energy columns are zero (not measured), so there is no energy comparison;
- DL latencies range from 0.97 ms (Transformer) to 8.7 ms (GRU), against 0.005 to 0.04 ms for the ML models, so a speed-up factor depends on which DL model is the reference.

Mean F1 over the datasets: GRU 0.890, Transformer 0.881, logistic regression 0.878, LightGBM 0.875.

## 6. Smaller points

- **Notebook 03 is not included.** `config/run_config_nb03.json`, the summary and the results are; notebooks 04 to 07 load its models.
- **Run counts.** The ML pipeline's saved files contain 120 centralised, 120 distributed, 1,440 robustness, 360 transfer and 60 RUL runs: 2,100 in total.
- **Samples are cycles.** Neighbouring cycles of one engine are strongly dependent; the engine-level split prevents leakage across engines, but row-level metrics are not independent observations.
- **Thresholds** are tuned on validation B per model and seed (`results/by_notebook/nb03_centralized/threshold_summary.csv`).
- **Figures quoted elsewhere.** Some earlier working reports quote values that differ from these files (for example 0.874 "at 80% node loss", which is a DL figure). Use `results/SUMMARY.md`.
