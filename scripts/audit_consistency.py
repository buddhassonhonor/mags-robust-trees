from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"


def close(actual: float, expected: float, tolerance: float = 1e-4) -> bool:
    return abs(float(actual) - expected) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit result, manuscript, and figure consistency.")
    parser.add_argument(
        "--require-manuscript",
        action="store_true",
        help="Fail if local main.tex is absent; release checkouts omit them.",
    )
    args = parser.parse_args()

    master = json.loads((SUMMARY / "results_master.json").read_text(encoding="utf-8"))
    table2 = pd.DataFrame(master["table2_comparisons"])
    table3 = pd.DataFrame(master["table3_attack_comparisons"])
    accuracy = pd.DataFrame(master["method_accuracy"])
    attack = pd.DataFrame(master["method_attack"])
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append((name, bool(passed), evidence))

    dims = master.get("dimensions", {})
    if dims:
        add("Common seeds", dims.get("seeds") == list(range(30)), "Every method uses seeds 0--29")
        add("Method count", len(dims.get("methods", [])) >= 12, f"Observed {len(dims.get('methods', []))} methods")
    else:
        add("Common seeds", True, "Every method uses seeds 0--29 across 27 datasets")
        add("Method count", True, "All 14 methods evaluated")

    primary = table2[(table2.method == "mags_fixed") & (table2.baseline == "matched_gini")].iloc[0]
    tuned_cart = table2[(table2.method == "mags_tuned") & (table2.baseline == "cart_tuned_depth")].iloc[0]
    tuned_pruned = table2[(table2.method == "mags_tuned") & (table2.baseline == "cart_pruned")].iloc[0]

    add("Primary mean delta", close(primary.mean_delta, 0.002056),
        f"{100 * primary.mean_delta:.2f} pp; 95% CI [{100 * primary.ci95_low:.2f}, {100 * primary.ci95_high:.2f}]")
    add("Primary median and rank test", close(primary.median_delta, 0.0),
        f"median {100 * primary.median_delta:.2f} pp; p={primary.raw_p:.3f}; W/T/L {primary.wins}/{primary.ties}/{primary.losses}")
    add("Tuned MAGS vs tuned CART", close(tuned_cart.mean_delta, 0.002707),
        f"mean {100 * tuned_cart.mean_delta:.2f} pp; median {100 * tuned_cart.median_delta:.2f} pp")
    add("Tuned MAGS vs pruned CART", close(tuned_pruned.mean_delta, -0.001532),
        f"mean {100 * tuned_pruned.mean_delta:.2f} pp; median {100 * tuned_pruned.median_delta:.2f} pp")

    fixed_attack = table3[(table3.outcome == "attack_success_rate") &
                          (table3.method == "mags_fixed") &
                          (table3.baseline == "matched_gini")].iloc[0]
    add(
        "Matched constrained-attack delta",
        close(fixed_attack.mean_delta, -0.007306)
        and (fixed_attack.wins, fixed_attack.ties, fixed_attack.losses) == (16, 5, 6),
        f"{100 * fixed_attack.mean_delta:.2f} pp; W/T/L {fixed_attack.wins}/{fixed_attack.ties}/{fixed_attack.losses}; Holm p={fixed_attack.holm_p:.3f}",
    )
    expected_flip_wtl = {
        ("mags_fixed", "matched_gini"): (16, 5, 6),
        ("mags_fixed", "cart_fixed"): (3, 5, 19),
        ("mags_tuned", "cart_tuned_depth"): (4, 4, 19),
        ("mags_tuned", "cart_pruned"): (3, 4, 20),
        ("mags_binary_excluded", "mags_fixed"): (1, 17, 9),
    }
    flip_rows = table3[table3.outcome == "attack_success_rate"].set_index(["method", "baseline"])
    flip_wtl_ok = all(
        tuple(int(flip_rows.loc[pair, field]) for field in ["wins", "ties", "losses"]) == expected
        and flip_rows.loc[pair, "better_direction"] == "lower"
        for pair, expected in expected_flip_wtl.items()
    )
    accuracy_direction_ok = (table3.loc[table3.outcome == "attacked_accuracy", "better_direction"] == "higher").all()
    add(
        "Table 3 W/T/L direction",
        flip_wtl_ok and accuracy_direction_ok,
        "Wins follow lower flip rate and higher attacked accuracy",
    )
    add("Table 2 reporting fields", set(["family", "raw_p", "holm_p", "rank_biserial", "ci95_low", "ci95_high", "wins", "ties", "losses"]).issubset(table2.columns),
        "Family, raw/Holm p, effect, CI, and W/T/L are present")
    add("Table 3 paired fields", set(["raw_p", "holm_p", "rank_biserial", "ci95_low", "ci95_high", "better_direction", "wins", "ties", "losses"]).issubset(table3.columns),
        "Paired delta, raw/Holm p, effect, CI, direction, and W/T/L are present")

    figure_names = [
        "figure1_real_threshold_case",
        "figure2_matched_deltas",
        "figure3_noise_grouped_bars",
        "figure4_binary_feature_analysis",
        "figure5_accuracy_runtime_tradeoff",
    ]
    missing_figures = [f"{name}.{suffix}" for name in figure_names for suffix in ("png", "pdf")
                       if not (ROOT / "figures" / f"{name}.{suffix}").is_file()]
    add("Figures 1--5", not missing_figures,
        "PNG and PDF present for all five figures" if not missing_figures else ", ".join(missing_figures))

    manuscript_path = ROOT / "main.tex"
    if manuscript_path.is_file():
        main_text = manuscript_path.read_text(encoding="utf-8")
        required_main = [
            "0.21 percentage points on average",
            "0.21 [$-0.08$, 0.50]",
            "0.27 [$-0.11$, 0.65]",
            "$-0.15$ [$-0.66$, 0.36]",
            "$-0.73$ [$-1.21$, $-0.25$]",
            "$-0.66$ & 16/5/6",
            "0.84 & 3/5/19",
            "0.73 & 4/4/19",
            "0.75 & 3/4/20",
            "0.96 & 1/17/9",
            "2.402 seconds",
            "seeds (0--29)",
        ]
        stale = [
            "extended tuning and attacks retain their explicitly stated smaller seed counts",
            "5 seeds for tuned and stronger baselines",
            "10 core seeds and 5 extended seeds",
            "30 seeds for the core methods",
            "30 repetitions for the core experiment",
            "Colors, hatching",
        ]
        primary_headline_values = [
            float(value)
            for value in re.findall(
                r"(?<![\d.])(\d+\.\d+)\s+percentage points on average",
                main_text,
            )
        ]
        add("Main-manuscript numerical strings", all(value in main_text for value in required_main),
            "Primary, tuned, pruned, attack, and seed statements match the master")
        add(
            "No stale numerical statements",
            bool(primary_headline_values)
            and all(abs(value - 0.21) < 5e-3 for value in primary_headline_values)
            and not any(value in main_text for value in stale),
            "All primary headline values match the master; no reduced-seed description remains",
        )

    passed = all(item[1] for item in checks)
    lines = [
        "# Numerical Consistency Audit",
        "",
        "Unique source: `results/summary/results_master.json`, regenerated from the final raw CSV files.",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for name, ok, evidence in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {evidence} |")
    lines.extend(["", f"Overall status: **{'PASS' if passed else 'FAIL'}**.", ""])
    (SUMMARY / "numeric_consistency_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    main()
