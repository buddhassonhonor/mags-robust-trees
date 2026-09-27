from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp" / "final_validation"
RAW = ROOT / "results" / "raw"
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


def make_figure1_classaware() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "exp"))
    sys.path.insert(0, str(ROOT / "src" / "harnesses"))
    from experiment_components import DATASETS
    from run_experiments import fit_custom, load_raw_dataset, preprocess_train_test, spec_for

    spec = spec_for("sonar")
    X_raw, y_raw, meta = load_raw_dataset(spec, seed=7)
    X_train_raw, X_test_raw, y_train, _ = train_test_split(X_raw, y_raw, test_size=0.35, random_state=7, stratify=y_raw)
    X_train, _, binary, meta = preprocess_train_test(X_train_raw, X_test_raw, meta)

    # Seed 7 shows an excluded same-class interval at the root.
    j_same = 10
    vals_same = X_train[:, j_same]
    order_same = np.argsort(vals_same)
    v_s = vals_same[order_same]
    y_s = y_train[order_same]

    # Find a wide same-class boundary in feature 10
    same_class_threshold = None
    same_gap = 0.0
    for k in range(len(v_s) - 1):
        if v_s[k+1] - v_s[k] > 1e-6 and y_s[k] == y_s[k+1] and y_s[k] == 0:
            gap = v_s[k+1] - v_s[k]
            if gap > same_gap:
                same_gap = gap
                same_class_threshold = (v_s[k] + v_s[k+1]) / 2.0
    # Seed 13 has a root split on the same feature but different thresholds.
    X_raw_b, y_raw_b, meta_b = load_raw_dataset(spec, seed=13)
    X_train_b, X_test_b, y_train_b, _ = train_test_split(
        X_raw_b, y_raw_b, test_size=0.35, random_state=13, stratify=y_raw_b
    )
    X_train_b, _, binary_b, _ = preprocess_train_test(X_train_b, X_test_b, meta_b)
    gini_tree, _ = fit_custom(X_train_b, y_train_b, spec, "matched_gini", 0.0, binary_b)
    mags_tree, _ = fit_custom(X_train_b, y_train_b, spec, "mags_fixed", 0.5, binary_b)
    assert gini_tree.tree["feature_idx"] == mags_tree.tree["feature_idx"]
    j_trans = int(gini_tree.tree["feature_idx"])
    gini_th = float(gini_tree.tree["threshold"])
    mags_th = float(mags_tree.tree["threshold"])
    assert not np.isclose(gini_th, mags_th)
    vals_trans = X_train_b[:, j_trans]
    order_trans = np.argsort(vals_trans)
    v_t = vals_trans[order_trans]
    y_t = y_train_b[order_trans]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
    rng = np.random.default_rng(2026)

    # Panel A: Why unrestricted gap preference is problematic
    ax0 = axes[0]
    jitter_s = rng.normal(0, 0.04, len(y_s))
    ax0.scatter(v_s, y_s + jitter_s, c=y_s, cmap="coolwarm", s=45, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax0.set_yticks([0, 1], ["Class 0", "Class 1"])
    ax0.set_title("(a) Boundary-point restriction excludes a same-class interval")
    ax0.set_xlabel(f"Standardized feature {j_same} (Sonar root, seed 7)")

    if same_class_threshold is not None:
        ax0.axvline(same_class_threshold, color="#d7191c", linestyle="--", linewidth=2.5, label="Excluded candidate ($y_k = y_{k+1} = 0$)")
        ax0.annotate(
            "Wide same-class interval\n(excluded candidate)",
            xy=(same_class_threshold, 0.05),
            xytext=(same_class_threshold + 0.15, 0.45),
            arrowprops=dict(facecolor="#d7191c", shrink=0.08, width=1.5, headwidth=8),
            fontsize=15,
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffeded", ec="#d7191c", lw=1),
        )
    ax0.legend(loc="upper right")

    # Panel B: Class-aware MAGS among valid class-transition candidates
    ax1 = axes[1]
    jitter_t = rng.normal(0, 0.04, len(y_t))
    ax1.scatter(v_t, y_t + jitter_t, c=y_t, cmap="coolwarm", s=45, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_yticks([0, 1], ["Class 0", "Class 1"])
    ax1.set_title("(b) Gap reward changes ranking among retained transitions")
    ax1.set_xlabel(f"Standardized feature {j_trans} (Sonar root, seed 13)")

    # Mark Gini threshold and MAGS threshold on valid transitions
    ax1.axvline(gini_th, color="#777777", linestyle="--", linewidth=2.5, label=f"Matched Gini threshold ({gini_th:.2f})")
    ax1.axvline(mags_th, color="#2b83ba", linestyle="-", linewidth=2.5, label=f"MAGS threshold ({mags_th:.2f})")
    ax1.annotate(
        "Valid class transition\n(Class 0 $\\leftrightarrow$ Class 1)",
        xy=(mags_th, 0.95),
        xytext=(mags_th - 0.75, 0.65),
        arrowprops=dict(facecolor="#2b83ba", shrink=0.08, width=1.5, headwidth=8),
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.3", fc="#edf6fc", ec="#2b83ba", lw=1),
    )
    ax1.legend(loc="upper right")

    save(fig, "figure1_real_threshold_case")
    print("Generated Figure 1 (Panel A and Panel B, no embedded table)")


def main_effect(df: pd.DataFrame) -> None:
    sub = df[(df.noise == 0.2) & df.method.isin(["mags_fixed", "matched_gini"])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    paired["delta"] = (paired.mags_fixed - paired.matched_gini) * 100.0  # percentage points
    means = paired.delta.groupby(level="dataset").mean().sort_values()
    cis = paired.delta.groupby(level="dataset").apply(ci95).reindex(means.index)
    colors = np.where(means >= 0, "#2b83ba", "#d7191c")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    y = np.arange(len(means))
    ax.barh(y, means.values, xerr=cis.values, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, [x.replace("_", " ") for x in means.index])
    ax.set_xlabel("Accuracy delta: fixed MAGS minus matched Gini (pp)")
    ax.set_title("Matched effect at Gaussian noise $\\sigma=0.2$")
    ax.text(0.98, 0.03, "30 paired splits per configuration", transform=ax.transAxes, ha="right", va="bottom", fontsize=16)
    save(fig, "figure2_matched_deltas")
    print("Generated Figure 2")


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
    ax.set_ylabel("Mean accuracy across 27 benchmark configurations")
    ax.set_ylim(0.68, 0.84)
    ax.set_title("Continuous-feature perturbation curves")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "figure3_noise_grouped_bars")
    print("Generated Figure 3")


def representation(stats: pd.DataFrame, df: pd.DataFrame) -> None:
    methods = ["matched_gini", "mags_fixed", "mags_binary_excluded"]
    labels = ["Matched Gini", "Fixed MAGS", "Binary-excluded"]
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
    axes[0].set_xticks(np.arange(3), labels, rotation=15, ha="right")
    axes[0].set_ylabel("Fraction of internal splits on binary columns")
    axes[0].set_title("Split-type frequency")
    axes[1].scatter(joined.binary, joined.delta * 100.0, s=60, color="#2b83ba", edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Fraction of encoded binary columns")
    axes[1].set_ylabel("MAGS minus matched Gini accuracy (pp)")
    axes[1].set_title(f"Binary association: $\\rho={rho:.2f}$, $p={p:.3f}$")
    save(fig, "figure4_binary_feature_analysis")
    print("Generated Figure 4")


def tradeoff(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    methods = [
        ("matched_gini", "Matched Gini", "o", "#777777"),
        ("mags_fixed", "Fixed MAGS", "o", "#2b83ba"),
        ("mags_tuned", "Tuned MAGS", "o", "#08519c"),
        ("cart_tuned_depth", "Tuned-depth CART", "o", "#fdae61"),
        ("cart_pruned", "Pruned CART", "o", "#d7191c"),
        ("random_forest", "Random Forest", "s", "#31a354"),
        ("extra_trees", "ExtraTrees", "s", "#74c476"),
        ("linear_svm", "Linear SVM", "s", "#756bb1"),
        ("rbf_svm", "RBF SVM", "s", "#bcbddc"),
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    for method, label, marker, color in methods:
        acc = df[(df.method == method) & (df.noise == 0.2)].groupby("dataset").accuracy.mean().mean()
        time_sec = stats[stats.method == method].groupby("dataset").fit_time_sec.mean().mean()
        ax.scatter(time_sec, acc, s=120, marker=marker, color=color, label=label, edgecolor="black", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit time per dataset (seconds, log scale)")
    ax.set_ylabel("Mean accuracy across 27 configurations at $\\sigma=0.2$")
    ax.set_title("Accuracy versus training time trade-off")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    save(fig, "figure5_accuracy_runtime_tradeoff")
    print("Generated Figure 5")


def main() -> None:
    df = pd.read_csv(RAW / "results.csv")
    stats = pd.read_csv(RAW / "tree_stats.csv")
    make_figure1_classaware()
    main_effect(df)
    noise_bars(df)
    representation(stats, df)
    tradeoff(df, stats)


if __name__ == "__main__":
    main()
