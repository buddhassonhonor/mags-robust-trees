#!/usr/bin/env python
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t, wilcoxon, spearmanr

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "exp" / "final_validation"
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"

RAW.mkdir(parents=True, exist_ok=True)
SUMMARY.mkdir(parents=True, exist_ok=True)

def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    x = values.dropna().astype(float)
    m = float(x.mean())
    if len(x) < 2:
        return m, m, m
    half = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - half, m + half

def rank_biserial(differences: pd.Series) -> float:
    x = differences.to_numpy(dtype=float)
    x = x[~np.isclose(x, 0.0, atol=1e-12)]
    if len(x) == 0:
        return 0.0
    ranks = rankdata(np.abs(x))
    pos = float(ranks[x > 0].sum())
    neg = float(ranks[x < 0].sum())
    return (pos - neg) / (pos + neg)

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

def paired_accuracy(metrics: pd.DataFrame, ours: str, baseline: str, family: str, noise: float = 0.2) -> dict:
    sub = metrics[(metrics.noise == noise) & metrics.method.isin([ours, baseline])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    differences = (paired[ours] - paired[baseline]).groupby(level="dataset").mean()
    differences.loc[np.isclose(differences, 0.0, atol=1e-12)] = 0.0
    m, low, high = mean_ci(differences)
    raw_p = float(wilcoxon(differences, alternative="two-sided").pvalue) if np.any(differences != 0) else 1.0
    return {
        "family": family,
        "outcome": f"accuracy_at_sigma_{noise}",
        "method": ours,
        "baseline": baseline,
        "datasets": len(differences),
        "seeds": int(paired.index.get_level_values("seed").nunique()),
        "mean_delta": m,
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
    m, low, high = mean_ci(differences)
    raw_p = float(wilcoxon(differences, alternative="two-sided").pvalue) if np.any(differences != 0) else 1.0
    better_direction = "lower" if outcome == "attack_success_rate" else "higher"
    favorable_differences = -differences if better_direction == "lower" else differences
    return {
        "family": family,
        "outcome": outcome,
        "method": ours,
        "baseline": baseline,
        "datasets": len(differences),
        "seeds": int(paired.index.get_level_values("seed").nunique()),
        "mean_delta": m,
        "ci95_low": low,
        "ci95_high": high,
        "median_delta": float(differences.median()),
        "rank_biserial": rank_biserial(differences),
        "better_direction": better_direction,
        "wins": int((favorable_differences > 0).sum()),
        "ties": int((favorable_differences == 0).sum()),
        "losses": int((favorable_differences < 0).sum()),
        "raw_p": raw_p,
    }

def main():
    for name in ["results.csv", "tree_stats.csv", "exact_attacks.csv"]:
        shutil.copy2(EXP_DIR / name, RAW / name)

    metrics = pd.read_csv(RAW / "results.csv")
    stats = pd.read_csv(RAW / "tree_stats.csv")
    attacks = pd.read_csv(RAW / "exact_attacks.csv")

    # Table 2: Primary comparisons & Strong Baselines
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
    table2.to_csv(ROOT / "classaware_primary_results.csv", index=False)

    # P4: Reviewer-fix ablation table
    ablation_specs = [
        ("matched_gini_unrestricted", "matched_gini", "ablation_filter", "Unrestricted Gini vs Class-aware Gini"),
        ("mags_fixed_unrestricted", "mags_fixed", "ablation_filter", "Unrestricted MAGS vs Class-aware MAGS"),
        ("mags_fixed", "matched_gini", "ablation_filter", "Class-aware MAGS vs Class-aware Gini (Primary)"),
        ("mags_fixed_unrestricted", "matched_gini_unrestricted", "ablation_filter", "Unrestricted MAGS vs Unrestricted Gini (Old)"),
    ]
    ablation_rows = []
    for ours, base, fam, desc in ablation_specs:
        clean = paired_accuracy(metrics, ours, base, fam, noise=0.0)
        noisy = paired_accuracy(metrics, ours, base, fam, noise=0.2)
        att_succ = paired_attack(attacks, ours, base, "attack_success_rate", fam)
        ablation_rows.append({
            "comparison": desc,
            "method": ours,
            "baseline": base,
            "clean_mean_delta": clean["mean_delta"] * 100.0,
            "clean_p": clean["raw_p"],
            "sigma02_mean_delta": noisy["mean_delta"] * 100.0,
            "sigma02_ci_low": noisy["ci95_low"] * 100.0,
            "sigma02_ci_high": noisy["ci95_high"] * 100.0,
            "sigma02_median_delta": noisy["median_delta"] * 100.0,
            "sigma02_W_T_L": f"{noisy['wins']}/{noisy['ties']}/{noisy['losses']}",
            "sigma02_p": noisy["raw_p"],
            "attack_succ_delta": att_succ["mean_delta"] * 100.0,
            "attack_succ_p": att_succ["raw_p"],
        })
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(SUMMARY / "classaware_ablation_results.csv", index=False)
    df_ablation.to_csv(ROOT / "classaware_ablation_results.csv", index=False)

    # Table 3: Attacks
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
    table3.to_csv(ROOT / "updated_attack_results.csv", index=False)

    # Method Attack Summaries
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

    # Method Accuracy & Runtime Summaries
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
    accuracy.to_csv(ROOT / "updated_runtime_results.csv", index=False)

    # Binary split fraction analysis
    bin_methods = ["matched_gini", "mags_fixed", "mags_binary_excluded", "matched_gini_unrestricted", "mags_fixed_unrestricted"]
    bin_rows = []
    for m in bin_methods:
        s = stats[stats.method == m].groupby("dataset").binary_split_fraction.mean()
        bin_rows.append({
            "method": m,
            "mean_binary_split_fraction": float(s.mean()),
            "ci95_low": float(s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s))),
            "ci95_high": float(s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))),
        })
    df_bin = pd.DataFrame(bin_rows)
    df_bin.to_csv(ROOT / "updated_binary_analysis.csv", index=False)

    # Spearman correlation between dataset binary fraction and MAGS delta
    sub = metrics[(metrics.noise == 0.2) & metrics.method.isin(["mags_fixed", "matched_gini"])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    delta = (paired.mags_fixed - paired.matched_gini).groupby(level="dataset").mean()
    delta.loc[np.isclose(delta, 0.0, atol=1e-12)] = 0.0
    bin_frac = metrics[(metrics.noise == 0.2) & (metrics.method == "mags_fixed")].groupby("dataset").binary_feature_fraction.mean()
    joined = pd.concat([bin_frac.rename("binary"), delta.rename("delta")], axis=1).dropna()
    rho, spearman_p = spearmanr(joined.binary, joined.delta)

    # Master summary JSON
    master = {
        "schema_version": "2.0_class_aware",
        "analysis_note": "Primary comparisons use the established boundary-point candidate set for both MAGS and matched Gini. The 27 benchmark configurations include one synthetic task and two sources of German Credit; all use seeds 0--29.",
        "dimensions": {
            "datasets": sorted(metrics.dataset.unique().tolist()),
            "seeds": sorted(int(seed) for seed in metrics.seed.unique()),
            "methods": sorted(metrics.method.unique().tolist()),
            "noise_levels": sorted(float(noise) for noise in metrics.noise.unique()),
        },
        "table2_comparisons": table2.to_dict(orient="records"),
        "table3_attack_comparisons": table3.to_dict(orient="records"),
        "ablation_comparisons": df_ablation.to_dict(orient="records"),
        "method_accuracy": accuracy.to_dict(orient="records"),
        "method_attack": attack_methods.to_dict(orient="records"),
        "binary_analysis": {
            "fractions": df_bin.to_dict(orient="records"),
            "spearman_rho": float(rho),
            "spearman_p": float(spearman_p),
        }
    }
    (SUMMARY / "results_master.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    (ROOT / "manuscript_results_summary.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    print("Master statistics built successfully!")

if __name__ == "__main__":
    main()
