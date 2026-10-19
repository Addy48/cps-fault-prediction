# Limitations

What the saved results can and cannot support. Each item was checked against the files in `results/` and the notebooks. Read it before quoting a number.

## 1. ML robustness covers centralised models only

Notebook 05 perturbs the test features and scores the six centralised models. Notebook 04 builds the distributed system but never perturbs it. So:

- there is no ML node-dropout experiment (node dropout means removing nodes from the vote at inference);
- there is no ML *replicated centralised* baseline (five replicas with different seeds and the same aggregation), which the protocol requires so that distributed and centralised are compared fairly;
- the node-dropout rows in `results/master/master_robustness_results.csv` all belong to the DL baselines (LSTM, GRU, TCN, Transformer), not to the ML pipeline.

The question the study sets out to answer, whether distributing the ML models makes them more robust, is therefore untested. What is tested is that distribution costs 0.013 to 0.032 F1 on clean data.

## 2. The ML perturbation results are almost certainly an artefact

Clean mean F1 over the six ML models is 0.867. Under Gaussian noise it is about 0.13 at every level (0.1, 0.2 and 0.3), which is what a model that predicts the positive class for every sample would give. The other perturbations also drop F1 to 0.23 to 0.42.

Cause (from `notebooks/05_robustness_testing.ipynb`): perturbations act on the 342-column engineered feature matrix. Noise has standard deviation `level * X.std().mean()`, a single scale computed over columns whose ranges differ widely. "Sensor failure" zeroes 10 to 30% of *feature columns* (not sensors), and "corrupted sensors" replaces columns with values beyond their maximum.

The DL baselines, which keep F1 at 0.81, 0.74 and 0.69 under Gaussian noise at the same three levels, were perturbed differently. The two are not comparable, and the protocol's rule that every pipeline faces identical perturbations was not met. The missing-value levels also differ: the protocol says 5, 10 and 20%; the ML run used 10, 20 and 30%, the DL run 5, 10 and 20%.

To get a usable noise result, perturb the raw sensor readings, recompute the features, and scale noise per signal.

## 3. The seeds do not measure variability

The split is fixed (`random_state = 42`). Seeds 42, 52, 62, 72 and 82 change only the model's internal randomness. Logistic regression, SVM, XGBoost and LightGBM are deterministic here, so all five seeds give the same F1 (standard deviation 0.000). Only random forest and CatBoost vary, by 0.003 to 0.029. Reported "mean ± std over seeds" therefore understates the real uncertainty. A fair estimate needs different engine splits, for example repeated or grouped cross-validation over engines.

The test sets are also small: 16 engines (FD001, FD003) and about 40 (FD002, FD004).

## 4. The statistical tests are underpowered

`results/statistical_tests/` compares models across four datasets:

- the Friedman test (chi-square 14.33, p = 0.046) has four blocks, one per dataset;
- the pairwise Wilcoxon tests have n = 4, so the smallest attainable p-value is 0.125 and none can be significant;
- the average-rank table includes an entry called "Other" (rank 7.0) that is not a model, and merges the four DL models into one "Deep Learning" entry.

Treat the ranking as descriptive, and do not describe any model difference as statistically significant on this evidence.

## 5. ML and DL comparisons mix pipelines

The DL baselines come from a separate pipeline whose code is not in this repository. Differences that affect comparisons:

- DL seeds were 42, 123, 256, 512 and 1024, not the protocol's five;
- DL latency and ML latency were measured separately, and the DL FLOPs and energy columns are zero (not measured), so there is no energy comparison;
- a "1000x faster" style claim depends on which DL model is the reference: the DL latencies range from 0.97 ms (Transformer) to 8.7 ms (GRU), against 0.005 to 0.04 ms for the ML models.

Mean F1 over the datasets is GRU 0.890, Transformer 0.881, logistic regression 0.878, LightGBM 0.875. The difference between the best DL and best ML model is 0.012, with no test that could resolve it.

## 6. Smaller points

- **Notebook 03 is missing.** `NB-3-AGAIN-outputs/nb-3-again.xpynb` on the author's machine is empty. Its configuration (`config/run_config_nb03.json`), summary and results are included, and notebooks 04 to 07 load its models.
- **Run counts.** The Kaggle summaries and earlier notes quote 2,100 to 2,580 runs. The saved files contain 120 centralised, 120 distributed, 1,440 robustness, 360 transfer and 60 RUL runs for the ML pipeline: 2,100.
- **Labels are rows.** A "sample" is one engine cycle, and neighbouring cycles of an engine are strongly dependent. The engine-level split prevents leakage across engines but row-level metrics are not independent observations.
- **Threshold choice.** Classification thresholds are tuned on validation B for each model and seed (`results/by_notebook/nb03_centralized/threshold_summary.csv`).
- **Reports in the author's folder** quote some numbers that disagree with the result files (for example F1 0.874 "at 80% node loss", which is a DL figure, and 0.947 versus 0.9468). Use `results/SUMMARY.md`.
