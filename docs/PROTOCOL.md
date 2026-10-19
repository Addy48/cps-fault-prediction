# Experiment protocol

The common protocol the ML and DL pipelines were meant to follow, and where each rule stands in the saved results. It is condensed from the project's working protocol document.

| # | Rule | Status for the ML pipeline |
|---|---|---|
| 1 | NASA C-MAPSS only, FD001 to FD004 | Met |
| 2 | Split by engine, never by row. Train 70%, validation A 10% (tuning), validation B 5% (thresholds), test 15% (final evaluation only) | Met (see `config/split_metadata.json`) |
| 3 | Fixed seed 42 for splitting, clustering and perturbations | Met. Consequence: the split never changes (see limitation 3) |
| 4 | No engine in more than one split | Met |
| 5 | Main horizon W = 20; W = 10 and 30 are optional | Met (W = 20 only) |
| 6 | Fault label 1 when RUL <= W | Met |
| 7 | Remove the same constant sensors in both pipelines | Met (six sensors removed) |
| 8 | Scaler fitted on training data only, then reused | Met |
| 9 | Five nodes with disjoint engines. Partitions: engine-ID block, and k-means (k = 5) on each engine's mean sensor vector over its first 20 cycles | Met |
| 10 | Centralised baselines: one pooled model, **and** a replicated baseline (five replicas, different seeds, same aggregation) | Pooled model met. Replicated baseline: not in the files here |
| 11 | Node dropout means removing nodes (or replicas) from the vote at inference; levels 10, 20, 40, 60, 80% | Not in the files here (DL only) |
| 12 | Aggregation: majority vote; weighted vote with weights from validation-A F1 only; confidence vote on averaged probabilities | Met |
| 13 | Perturbations identical for every pipeline: Gaussian noise 0.1, 0.2, 0.3; missing values 5, 10, 20% (training-median imputation); sensor failure 10, 20, 30%; corrupted sensors | Differ: ML perturbs the engineered feature matrix and used 10, 20, 30% for missing values; DL differs (see limitation 2) |
| 14 | DL sequences must not cross engine boundaries | DL side, not checked here |
| 15 | At least five seeds; report mean ± std | Five seeds run; they do not vary the split (limitation 3) |
| 16 | Wilcoxon signed-rank and paired t-test for ML versus DL and centralised versus distributed | Partly: Wilcoxon over four datasets only (limitation 4) |
| 17 | Deployment: CPU latency, memory, model size, training and inference time (DL also FLOPs and parameters) | Met for ML; DL FLOPs and energy are zero in the files |
| 18 | Cross-dataset: FD001 to FD003, FD001 to FD004, FD002 to FD004, FD003 to FD004, plus multi-dataset training (FD001+FD003, all) | ML ran all twelve single-source pairs. Multi-dataset training is in the DL files only |
| 19 | Results as CSV with common columns, for example `model,dataset,setup,precision,recall,f1,auroc` | Partly: the two pipelines' CSVs use different column sets, merged in notebook 08 |

Splits in this run (engines per split):

| Dataset | Train | Validation A | Validation B | Test |
|---|---:|---:|---:|---:|
| FD001 | 70 | 9 | 5 | 16 |
| FD002 | 182 | 25 | 13 | 40 |
| FD003 | 70 | 9 | 5 | 16 |
| FD004 | 174 | 24 | 12 | 39 |
