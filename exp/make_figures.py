from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp" / "final_validation"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def ci95(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return 0.0
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=100, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main_effect(df: pd.DataFrame) -> None:
    sub = df[(df.noise == 0.2) & df.method.isin(["mags_fixed", "matched_gini"])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    paired["delta"] = paired.mags_fixed - paired.matched_gini
    means = paired.delta.groupby(level="dataset").mean().sort_values()
    cis = paired.delta.groupby(level="dataset").apply(ci95).reindex(means.index)
    colors = np.where(means >= 0, "#2b83ba", "#d7191c")
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(means))
    ax.barh(y, means.values, xerr=cis.values, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, [x.replace("_", " ") for x in means.index])
    ax.set_xlabel("Accuracy delta: fixed MAGS minus matched Gini")
    ax.set_title("Matched-depth effect at Gaussian noise $\\sigma=0.2$")
    ax.text(0.98, 0.03, "30 paired splits per dataset", transform=ax.transAxes, ha="right", va="bottom", fontsize=16)
    save(fig, "figure2_matched_deltas")


def noise_bars(df: pd.DataFrame) -> None:
    methods = ["matched_gini", "mags_fixed", "mags_tuned", "cart_tuned_depth", "cart_pruned"]
    labels = ["Matched Gini", "Fixed MAGS", "Tuned MAGS", "Tuned-depth CART", "Pruned CART"]
    colors = ["#777777", "#2b83ba", "#08519c", "#fdae61", "#d7191c"]
    x = np.arange(5)
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        rows = []
        errs = []
        for noise in sorted(df.noise.unique()):
            values = df[(df.method == method) & (df.noise == noise)].groupby("dataset").accuracy.mean()
            rows.append(values.mean())
            errs.append(ci95(values))
        ax.bar(x + (i - 2) * width, rows, width, yerr=errs, capsize=2, label=label, color=color)
    ax.set_xticks(x, [f"{z:.2f}" for z in sorted(df.noise.unique())])
    ax.set_xlabel("Gaussian noise standard deviation (continuous columns)")
    ax.set_ylabel("Mean accuracy across datasets")
    ax.set_ylim(0.68, 0.84)
    ax.set_title("Continuous-feature perturbation curve")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "figure3_noise_grouped_bars")


def representation(stats: pd.DataFrame, df: pd.DataFrame) -> None:
    methods = ["matched_gini", "mags_fixed", "mags_binary_excluded"]
    labels = ["Matched Gini", "Fixed MAGS", "Binary-excluded MAGS"]
    vals, errs = [], []
    for method in methods:
        x = stats[stats.method == method].groupby("dataset").binary_split_fraction.mean()
        vals.append(x.mean())
        errs.append(ci95(x))
    sub = df[(df.noise == 0.2) & df.method.isin(["mags_fixed", "matched_gini"])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    delta = (paired.mags_fixed - paired.matched_gini).groupby(level="dataset").mean()
    delta.loc[np.isclose(delta, 0.0, atol=1e-12)] = 0.0
    binary = df[(df.noise == 0.2) & (df.method == "mags_fixed")].groupby("dataset").binary_feature_fraction.mean()
    joined = pd.concat([binary.rename("binary"), delta.rename("delta")], axis=1).dropna()
    rho, p = spearmanr(joined.binary, joined.delta)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].bar(np.arange(3), vals, yerr=errs, capsize=4, color=["#777777", "#2b83ba", "#66c2a5"])
    axes[0].set_xticks(np.arange(3), labels, rotation=18, ha="right")
    axes[0].set_ylabel("Fraction of internal splits on binary columns")
    axes[0].set_title("Observed split-type preference")
    axes[1].scatter(joined.binary, joined.delta, s=60, color="#2b83ba", edgecolor="white")
    for name, row in joined.iterrows():
        if abs(row.delta) > 0.01:
            axes[1].annotate(name.replace("_", " "), (row.binary, row.delta), fontsize=12)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Fraction of encoded binary columns")
    axes[1].set_ylabel("MAGS minus matched-Gini accuracy")
    axes[1].set_title(f"Representation association: $\\rho={rho:.2f}$, $p={p:.3f}$")
    save(fig, "figure4_binary_feature_analysis")


def threshold_case() -> None:
    info = json.loads((EXP / "threshold_case.json").read_text(encoding="utf-8"))
    cand = pd.read_csv(EXP / "threshold_case_candidates.csv")
    pts = pd.read_csv(EXP / "threshold_case_points.csv")
    g = cand[cand.selected_score_alpha_0].iloc[0]
    m = cand[cand.selected_score_alpha_05].iloc[0]
    mo = cand[cand.selected_score_margin_only].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1.0, 1.0, 0.72]})
    rng = np.random.default_rng(2026)
    for ax, feature, title in [
        (axes[0], int(g.feature), "Gini tie resolved by adjacent-gap preference"),
        (axes[1], int(mo.feature), "Margin-only choice can sacrifice impurity gain"),
    ]:
        q = pts[pts.feature == feature]
        y = q["class"].to_numpy() + rng.normal(0, 0.035, len(q))
        ax.scatter(q.value, y, c=q["class"], cmap="coolwarm", s=35, alpha=0.8, edgecolor="none")
        ax.set_yticks(sorted(q["class"].unique()), [f"Class {c}" for c in sorted(q["class"].unique())])
        ax.set_title(title)
        ax.set_xlabel(f"Standardized encoded feature {feature}")
    axes[0].axvline(float(g.threshold), color="#777777", linestyle="--", linewidth=2, label=f"Gini: {g.threshold:.2f}")
    axes[0].axvline(float(m.threshold), color="#2b83ba", linewidth=2, label=f"MAGS $\\alpha=0.5$: {m.threshold:.2f}")
    axes[0].legend(loc="best")
    axes[1].axvline(float(mo.threshold), color="#d7191c", linewidth=2, label=f"Margin only: {mo.threshold:.2f}")
    axes[1].legend(loc="best")
    axes[2].axis("off")
    alpha_rows = []
    for label, column in [("0", "score_alpha_0"), ("0.1", "score_alpha_01"), ("0.3", "score_alpha_03"), ("0.5", "score_alpha_05"), ("0.7", "score_alpha_07"), ("1.0", "score_alpha_10"), ("margin only", "score_margin_only")]:
        row = cand.loc[cand[column].idxmax()]
        alpha_rows.append([label, int(row.feature), f"{row.threshold:.3f}", f"{row.normalized_gini_gain:.3f}", f"{row.normalized_adjacent_gap:.3f}"])
    table = axes[2].table(
        cellText=alpha_rows,
        colLabels=["$\\alpha$", "Feature", "Threshold", "Norm. gain", "Norm. gap"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.0, 1.35)
    axes[2].set_title("Selected candidate as the adjacent-gap weight changes", pad=4)
    fig.suptitle(f"Real {info['dataset'].replace('_', ' ').title()} node ({info['n_node']} training samples)", y=1.02)
    save(fig, "figure1_real_threshold_case")


def tradeoff(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    methods = ["matched_gini", "mags_fixed", "mags_tuned", "cart_tuned_depth", "cart_pruned", "random_forest", "extra_trees", "linear_svm", "rbf_svm"]
    labels = {
        "matched_gini": "Matched Gini", "mags_fixed": "Fixed MAGS", "mags_tuned": "Tuned MAGS",
        "cart_tuned_depth": "Tuned CART", "cart_pruned": "Pruned CART", "random_forest": "RF",
        "extra_trees": "ExtraTrees", "linear_svm": "Linear SVM", "rbf_svm": "RBF SVM",
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for method in methods:
        acc = df[(df.method == method) & (df.noise == 0.2)].groupby("dataset").accuracy.mean().mean()
        runtime = stats[stats.method == method].groupby("dataset").fit_time_sec.mean().mean()
        marker = "o" if method in ["matched_gini", "mags_fixed", "mags_tuned", "cart_tuned_depth", "cart_pruned"] else "s"
        ax.scatter(runtime, acc, s=100, marker=marker)
        ax.annotate(labels[method], (runtime, acc), xytext=(5, 5), textcoords="offset points", fontsize=13)
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit time per dataset-seed (seconds, log scale)")
    ax.set_ylabel("Mean accuracy at $\\sigma=0.2$")
    ax.set_title("Predictive accuracy, training time, and model class")
    ax.text(0.02, 0.03, "Circles: single trees; squares: ensembles/SVMs", transform=ax.transAxes, fontsize=15)
    save(fig, "figure5_accuracy_runtime_tradeoff")


def main() -> None:
    df = pd.read_csv(EXP / "results.csv")
    stats = pd.read_csv(EXP / "tree_stats.csv")
    main_effect(df)
    noise_bars(df)
    representation(stats, df)
    threshold_case()
    tradeoff(df, stats)
    print("Generated manuscript figures in", FIG)


if __name__ == "__main__":
    main()
