"""Summarize paired robust-tree results on distinct binary-class datasets."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import t, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"
EXCLUDED_DUPLICATE = "credit_g_openml"
METHODS = [
    "matched_gini", "mags_fixed", "cart_fixed", "cart_tuned_depth",
    "cart_pruned", "noise_augmented_cart", "groot", "chen_heuristic",
]


def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return [mean, mean - half, mean + half]


def main() -> None:
    robust = pd.read_csv(RAW / "robust_tree_comparison.csv")
    assert not robust.duplicated(["dataset", "seed", "method"]).any()
    datasets = sorted(set(robust.dataset) - {EXCLUDED_DUPLICATE})
    assert len(datasets) == 14
    robust = robust[robust.dataset.isin(datasets)]
    assert len(robust) == 14 * 30 * 2
    assert set(robust.groupby(["dataset", "method"]).seed.nunique()) == {30}

    metrics = pd.read_csv(RAW / "results.csv")
    attacks = pd.read_csv(RAW / "exact_attacks.csv")
    metrics = metrics[metrics.dataset.isin(datasets) & metrics.method.isin(METHODS)]
    attacks = attacks[attacks.dataset.isin(datasets) & attacks.method.isin(METHODS)]
    clean = metrics[np.isclose(metrics.noise, 0)].set_index(["dataset", "seed", "method"]).accuracy
    noisy = metrics[np.isclose(metrics.noise, 0.2)].set_index(["dataset", "seed", "method"]).accuracy
    attack = attacks.set_index(["dataset", "seed", "method"])

    rows = []
    for method in METHODS:
        if method in ("groot", "chen_heuristic"):
            sub = robust[robust.method == method].set_index(["dataset", "seed"])
            by_dataset = sub.groupby(level="dataset")[
                ["clean_accuracy", "noisy_accuracy", "attack_success_rate", "attacked_accuracy", "fit_time_sec"]
            ].mean()
        else:
            index = pd.MultiIndex.from_product([datasets, range(30), [method]], names=["dataset", "seed", "method"])
            frame = pd.DataFrame({
                "clean_accuracy": clean.reindex(index).to_numpy(),
                "noisy_accuracy": noisy.reindex(index).to_numpy(),
                "attack_success_rate": attack.attack_success_rate.reindex(index).to_numpy(),
                "attacked_accuracy": attack.attacked_accuracy.reindex(index).to_numpy(),
            }, index=index)
            assert not frame.isna().any().any(), method
            by_dataset = frame.groupby(level="dataset").mean()
        assert len(by_dataset) == len(datasets)
        for outcome in ("clean_accuracy", "noisy_accuracy", "attack_success_rate", "attacked_accuracy"):
            rows.append({"method": method, "outcome": outcome, "mean": float(by_dataset[outcome].mean()), "n_datasets": len(datasets), "n_seeds": 30})
    summary = pd.DataFrame(rows)

    robust_attacks = robust.set_index(["dataset", "seed", "method"])
    comparisons = []
    for method in ("groot", "chen_heuristic"):
        for outcome in ("attack_success_rate", "attacked_accuracy"):
            ours = robust_attacks[outcome].xs(method, level="method")
            base = attack[outcome].xs("mags_fixed", level="method")
            delta = (ours - base).groupby(level="dataset").mean().sort_index()
            assert len(delta) == len(datasets)
            mean, low, high = interval(delta.to_numpy())
            p = float(wilcoxon(delta).pvalue) if not np.allclose(delta, 0) else 1.0
            comparisons.append({
                "method": method,
                "baseline": "mags_fixed",
                "outcome": outcome,
                "delta": mean,
                "ci95_low": low,
                "ci95_high": high,
                "raw_p": p,
                "wins": int((delta < 0).sum()) if outcome == "attack_success_rate" else int((delta > 0).sum()),
                "ties": int(np.isclose(delta, 0).sum()),
                "losses": int((delta > 0).sum()) if outcome == "attack_success_rate" else int((delta < 0).sum()),
            })
    comparison_frame = pd.DataFrame(comparisons)
    for outcome, indices in comparison_frame.groupby("outcome").groups.items():
        order = comparison_frame.loc[indices, "raw_p"].sort_values()
        running = 0.0
        for rank, (idx, p) in enumerate(order.items()):
            running = max(running, min(1.0, (len(order) - rank) * float(p)))
            comparison_frame.loc[idx, "holm_p"] = running

    SUMMARY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY / "robust_tree_methods.csv", index=False)
    comparison_frame.to_csv(SUMMARY / "robust_tree_comparisons.csv", index=False)
    (SUMMARY / "robust_tree_protocol.json").write_text(json.dumps({
        "datasets": datasets,
        "excluded_duplicate": EXCLUDED_DUPLICATE,
        "seeds": list(range(30)),
        "methods": METHODS,
        "attack_epsilon": 0.05,
        "binary_features_immutable": True,
        "robust_tree_implementation": "groot-trees==0.0.17; chen_heuristic=True for Chen-style comparison",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "groot_trees": version("groot-trees"),
        },
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
