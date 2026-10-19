import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_summary", ROOT / "scripts" / "build_summary.py")
build_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_summary)
RES = ROOT / "results"


def test_run_counts_match_the_documented_design():
    counts = {
        "nb03_centralized/centralized_results.csv": 120,
        "nb04_distributed/distributed_results.csv": 120,
        "nb05_robustness/robustness_results.csv": 1440,
        "nb06_cross_dataset/cross_dataset_results.csv": 360,
        "nb07_rul/rul_results.csv": 60,
    }
    for path, n in counts.items():
        assert len(pd.read_csv(RES / "by_notebook" / path)) == n, path


def test_headline_numbers_in_the_readme():
    cen = pd.read_csv(RES / "by_notebook/nb03_centralized/centralized_results.csv")
    means = cen.groupby(["model", "dataset"]).f1.mean()
    assert round(means.max(), 3) == 0.947 and means.idxmax() == ("lightgbm", "FD003")
    by_model = cen.groupby(["model", "dataset"]).f1.mean().groupby("model").mean()
    assert round(by_model["logistic_regression"], 3) == 0.878
    assert round(by_model["lightgbm"], 3) == 0.875


def test_distributed_is_below_centralised_lightgbm():
    table = build_summary.distributed()
    assert (table["Gap"].astype(float) < 0).all()


def test_ml_robustness_collapse_is_recorded():
    rob = pd.read_csv(RES / "by_notebook/nb05_robustness/robustness_results.csv")
    noise = rob[rob.perturbation_type == "gaussian_noise"].f1.mean()
    assert 0.12 < noise < 0.14


def test_node_dropout_rows_are_dl_only():
    master = pd.read_csv(RES / "master/master_robustness_results.csv")
    nd = master[master.perturbation_type == "node_dropout"]
    assert set(nd.model) == {"LSTM", "GRU", "TCN", "Transformer"}


def test_fixed_split_makes_four_models_seed_invariant():
    cen = pd.read_csv(RES / "by_notebook/nb03_centralized/centralized_results.csv")
    std = cen.groupby(["model", "dataset"]).f1.std()
    for model in ("logistic_regression", "svm", "xgboost", "lightgbm"):
        assert std[model].max() < 1e-9
    assert std["random_forest"].max() > 0.001
