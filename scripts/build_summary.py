"""Rebuild the headline tables from the saved result files in results/.

    python scripts/build_summary.py        # writes results/SUMMARY.md and prints it

Nothing is re-trained. Every number comes from the per-notebook CSVs in
results/by_notebook/ (ML side) or results/dl_baselines/ (DL side). The ML and
DL results are kept apart: they were produced by different pipelines.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["logistic_regression", "svm", "random_forest", "xgboost", "lightgbm", "catboost"]


def pm(series: pd.Series, digits: int = 3) -> str:
    return f"{series.mean():.{digits}f} ± {series.std(ddof=1):.{digits}f}"


def load(*parts: str) -> pd.DataFrame:
    return pd.read_csv(RES.joinpath(*parts))


def centralized() -> pd.DataFrame:
    df = load("by_notebook", "nb03_centralized", "centralized_results.csv")
    rows = []
    for model in MODELS:
        row = {"Model": model}
        for ds in DATASETS:
            row[ds] = pm(df[(df.model == model) & (df.dataset == ds)].f1)
        row["Mean of datasets"] = f"{df[df.model == model].groupby('dataset').f1.mean().mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def distributed() -> pd.DataFrame:
    dist = load("by_notebook", "nb04_distributed", "distributed_results.csv")
    cen = load("by_notebook", "nb03_centralized", "centralized_results.csv")
    cen = cen[cen.model == "lightgbm"]
    rows = []
    for ds in DATASETS:
        sub = dist[dist.dataset == ds]
        grp = sub.groupby(["partition_method", "aggregation_method"]).f1.mean()
        best = grp.idxmax()
        best_runs = sub[(sub.partition_method == best[0]) & (sub.aggregation_method == best[1])].f1
        rows.append({
            "Dataset": ds,
            "Centralised LightGBM": pm(cen[cen.dataset == ds].f1),
            "Best distributed setting": f"{best[0]} + {best[1]}",
            "Best distributed F1": pm(best_runs),
            "Gap": f"{best_runs.mean() - cen[cen.dataset == ds].f1.mean():+.3f}",
            "Mean over all six settings": f"{grp.mean():.3f}",
        })
    return pd.DataFrame(rows)


def robustness() -> pd.DataFrame:
    rob = load("by_notebook", "nb05_robustness", "robustness_results.csv")
    clean = load("by_notebook", "nb03_centralized", "centralized_results.csv").f1.mean()
    rows = [{"Perturbation": "none (clean test data)", "Level": "-", "Mean F1, six ML models": f"{clean:.3f}"}]
    for ptype in ["gaussian_noise", "missing_values", "sensor_failure", "corrupted_sensors"]:
        for level, sub in rob[rob.perturbation_type == ptype].groupby("perturbation_level"):
            rows.append({"Perturbation": ptype, "Level": f"{level:g}", "Mean F1, six ML models": f"{sub.f1.mean():.3f}"})
    return pd.DataFrame(rows)


def cross_dataset() -> pd.DataFrame:
    df = load("by_notebook", "nb06_cross_dataset", "cross_dataset_results.csv")
    m = df.groupby(["train_dataset", "test_dataset"]).f1.mean().unstack().round(3)
    m.index.name = "Train \\ Test"
    return m.reset_index().fillna("")


def rul() -> pd.DataFrame:
    df = load("by_notebook", "nb07_rul", "rul_results.csv")
    rows = []
    for model in ["lightgbm", "random_forest", "xgboost"]:
        row = {"Model": model}
        for ds in DATASETS:
            row[f"{ds} RMSE"] = pm(df[(df.model == model) & (df.dataset == ds)].rmse, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def deployment() -> pd.DataFrame:
    df = load("by_notebook", "nb03_centralized", "deployment_metrics.csv")
    g = df.groupby("model").agg(latency_ms=("latency_ms", "median"), size_mb=("model_size_mb", "max"))
    ml = g.loc[MODELS].round(4).reset_index()
    dl = load("dl_baselines", "dl_deployment_metrics.csv")[["model", "cpu_latency_ms", "model_size_mb"]]
    dl.columns = ["model", "latency_ms", "size_mb"]
    out = pd.concat([ml, dl.round(4)], ignore_index=True)
    out.insert(1, "Family", ["ML"] * len(ml) + ["DL"] * len(dl))
    return out.rename(columns={"model": "Model", "latency_ms": "Latency (ms)", "size_mb": "Max size (MB)"})


def dl_vs_ml() -> pd.DataFrame:
    dl = load("dl_baselines", "centralized_dl_results.csv")
    cen = load("by_notebook", "nb03_centralized", "centralized_results.csv")
    rows = []
    for family, df, models in (("ML", cen, MODELS), ("DL", dl, ["LSTM", "GRU", "TCN", "Transformer"])):
        for model in models:
            per_ds = df[df.model == model].groupby("dataset").f1.mean()
            rows.append({"Family": family, "Model": model, **{ds: f"{per_ds[ds]:.3f}" for ds in DATASETS}, "Mean": f"{per_ds.mean():.3f}"})
    return pd.DataFrame(rows)


def main() -> None:
    sections = [
        ("Centralised classification (five seeds, F1 mean ± std)", centralized()),
        ("Five-node distributed LightGBM versus centralised LightGBM", distributed()),
        ("Robustness of the centralised ML models (perturbations on the 342-column feature matrix)", robustness()),
        ("Cross-dataset transfer, mean F1 over six ML models and five seeds", cross_dataset()),
        ("RUL regression (RMSE in cycles, mean ± std)", rul()),
        ("Deployment (median latency, largest model size)", deployment()),
        ("ML and DL baselines side by side (mean F1 per dataset; different pipelines, see docs/LIMITATIONS.md)", dl_vs_ml()),
    ]
    out = ["# Summary tables rebuilt from the saved results", "",
           "Generated by `scripts/build_summary.py`. Nothing was re-trained.", ""]
    for title, table in sections:
        out += [f"## {title}", "", table.to_markdown(index=False), ""]
    text = "\n".join(out)
    (RES / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
