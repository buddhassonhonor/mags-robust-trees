from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "exp" / "final_validation"
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    x = values.dropna().astype(float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def rank_biserial(differences: pd.Series) -> float:
    x = differences.to_numpy(dtype=float)
    x = x[~np.isclose(x, 0.0, atol=1e-12)]
    if len(x) == 0:
        return 0.0
    ranks = rankdata(np.abs(x))
    positive = float(ranks[x > 0].sum())
    negative = float(ranks[x < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["holm_p"] = np.nan
    for family, index in frame.groupby("family").groups.items():
        ordered = frame.loc[index, "raw_p"].sort_values()
        adjusted = []
        running = 0.0
        m = len(ordered)
        for rank, (idx, value) in enumerate(ordered.items()):
            running = max(running, min(1.0, (m - rank) * float(value)))
            adjusted.append((idx, running))
        for idx, value in adjusted:
            frame.loc[idx, "holm_p"] = value
    return frame


def paired_accuracy(metrics: pd.DataFrame, ours: str, baseline: str, family: str) -> dict:
    sub = metrics[(metrics.noise == 0.2) & metrics.method.isin([ours, baseline])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    differences = (paired[ours] - paired[baseline]).groupby(level="dataset").mean()
    differences.loc[np.isclose(differences, 0.0, atol=1e-12)] = 0.0
    mean, low, high = mean_ci(differences)
    raw_p = float(wilcoxon(differences, alternative="two-sided").pvalue) if np.any(differences != 0) else 1.0
    return {
        "family": family,
        "outcome": "accuracy_at_sigma_0.2",
        "method": ours,
        "baseline": baseline,
        "datasets": len(differences),
        "seeds": int(paired.index.get_level_values("seed").nunique()),
        "mean_delta": mean,
        "ci95_low": low,
        "ci95_high": high,
        "median_delta": float(differences.median()),
        "rank_biserial": rank_biserial(differences),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "raw_p": raw_p,
    }


def paired_attack(attacks: pd.DataFrame, ours: str, baseline: str, outcome: str, family: str) -> dict:
    sub = attacks[attacks.method.isin([ours, baseline])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values=outcome).dropna()
    differences = (paired[ours] - paired[baseline]).groupby(level="dataset").mean()
    differences.loc[np.isclose(differences, 0.0, atol=1e-12)] = 0.0
    mean, low, high = mean_ci(differences)
    raw_p = float(wilcoxon(differences, alternative="two-sided").pvalue) if np.any(differences != 0) else 1.0
    return {
        "family": family,
        "outcome": outcome,
        "method": ours,
        "baseline": baseline,
        "datasets": len(differences),
        "seeds": int(paired.index.get_level_values("seed").nunique()),
        "mean_delta": mean,
        "ci95_low": low,
        "ci95_high": high,
        "median_delta": float(differences.median()),
        "rank_biserial": rank_biserial(differences),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "raw_p": raw_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unique result master and final statistical tables.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    for name in ["results.csv", "tree_stats.csv", "exact_attacks.csv"]:
        shutil.copy2(source / name, RAW / name)
    for name in ["metadata.json"]:
        if (source / name).exists():
            payload = json.loads((source / name).read_text(encoding="utf-8"))
            if name == "metadata.json" and "python" in payload:
                payload["python"] = Path(payload["python"]).name
            (ROOT / "metadata" / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    versions = run_metadata.get("library_versions", {})
    reproducibility = {
        "experiment_timestamp": run_metadata.get("timestamp"),
        "python": run_metadata.get("python_version", "").split()[0],
        "numpy": versions.get("numpy"),
        "pandas": versions.get("pandas"),
        "scipy": versions.get("scipy"),
        "scikit_learn": versions.get("scikit_learn"),
        "operating_system": run_metadata.get("platform"),
        "core_protocol": "27 datasets; all 12 methods; identical stratified splits for seeds 0--29",
        "exact_attack_protocol": "27 datasets; all 8 single-tree methods; seeds 0--29; encoded binary columns fixed",
        "random_seeds": "0--29 for every method and constrained attack",
    }
    (ROOT / "metadata" / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2), encoding="utf-8"
    )

    metrics = pd.read_csv(RAW / "results.csv")
    stats = pd.read_csv(RAW / "tree_stats.csv")
    attacks = pd.read_csv(RAW / "exact_attacks.csv")

    table2_specs = [
        ("mags_fixed", "matched_gini", "primary"),
        ("mags_tuned", "cart_tuned_depth", "secondary_strong_baselines"),
        ("mags_tuned", "cart_pruned", "secondary_strong_baselines"),
        ("mags_tuned", "noise_augmented_cart", "secondary_strong_baselines"),
        ("mags_tuned", "random_forest", "secondary_strong_baselines"),
        ("mags_tuned", "extra_trees", "secondary_strong_baselines"),
        ("mags_tuned", "linear_svm", "secondary_strong_baselines"),
        ("mags_tuned", "rbf_svm", "secondary_strong_baselines"),
        ("mags_binary_excluded", "matched_gini", "exploratory_ablation"),
        ("mags_binary_excluded", "mags_fixed", "exploratory_ablation"),
    ]
    table2 = holm_adjust(pd.DataFrame([paired_accuracy(metrics, *spec) for spec in table2_specs]))
    table2.to_csv(SUMMARY / "table2_comparisons.csv", index=False)

    attack_pairs = [
        ("mags_fixed", "matched_gini"),
        ("mags_fixed", "cart_fixed"),
        ("mags_tuned", "cart_tuned_depth"),
        ("mags_tuned", "cart_pruned"),
        ("mags_binary_excluded", "mags_fixed"),
    ]
    table3_rows = []
    for outcome in ["attack_success_rate", "attacked_accuracy"]:
        family = f"constrained_attack_{outcome}"
        table3_rows.extend(paired_attack(attacks, ours, base, outcome, family) for ours, base in attack_pairs)
    table3 = holm_adjust(pd.DataFrame(table3_rows))
    table3.to_csv(SUMMARY / "table3_attack_comparisons.csv", index=False)

    attack_method_rows = []
    for method, group in attacks.groupby("method"):
        dataset_means = group.groupby("dataset")[["attack_success_rate", "attacked_accuracy", "median_min_flip_linf"]].mean()
        attack_method_rows.append({
            "method": method,
            "attack_success_rate": float(dataset_means.attack_success_rate.mean()),
            "attacked_accuracy": float(dataset_means.attacked_accuracy.mean()),
            "median_min_flip_linf": float(dataset_means.median_min_flip_linf.median()),
            "datasets": len(dataset_means),
            "seeds": group.seed.nunique(),
        })
    attack_methods = pd.DataFrame(attack_method_rows).sort_values("method")
    attack_methods.to_csv(SUMMARY / "method_attack.csv", index=False)

    accuracy_rows = []
    for method, group in metrics[metrics.noise == 0.2].groupby("method"):
        dataset_means = group.groupby("dataset").accuracy.mean()
        mean, low, high = mean_ci(dataset_means)
        clean_means = metrics[(metrics.noise == 0.0) & (metrics.method == method)].groupby("dataset").accuracy.mean()
        fit_means = stats[stats.method == method].groupby("dataset").fit_time_sec.mean()
        accuracy_rows.append({
            "method": method,
            "mean_accuracy": mean,
            "ci95_low": low,
            "ci95_high": high,
            "clean_accuracy": float(clean_means.mean()),
            "mean_fit_time_sec": float(fit_means.mean()),
            "datasets": len(dataset_means),
            "seeds": group.seed.nunique(),
        })
    accuracy = pd.DataFrame(accuracy_rows).sort_values("method")
    accuracy.to_csv(SUMMARY / "method_accuracy.csv", index=False)

    master = {
        "schema_version": "1.0",
        "analysis_note": "All methods use seeds 0--29 and identical dataset splits. Exact attacks are exact under the stated continuous-coordinate and fixed-binary constraints.",
        "dimensions": {
            "datasets": sorted(metrics.dataset.unique().tolist()),
            "seeds": sorted(int(x) for x in metrics.seed.unique()),
            "methods": sorted(metrics.method.unique().tolist()),
            "noise_levels": sorted(float(x) for x in metrics.noise.unique()),
        },
        "table2_comparisons": table2.to_dict(orient="records"),
        "table3_attack_comparisons": table3.to_dict(orient="records"),
        "method_accuracy": accuracy.to_dict(orient="records"),
        "method_attack": attack_methods.to_dict(orient="records"),
        "raw_files": {
            "metrics": "../raw/results.csv",
            "tree_statistics": "../raw/tree_stats.csv",
            "constrained_exact_attacks": "../raw/exact_attacks.csv",
        },
    }
    (SUMMARY / "results_master.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY / 'results_master.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
