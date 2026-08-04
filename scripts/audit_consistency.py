from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"


def close(actual: float, expected: float, tolerance: float = 5e-12) -> bool:
    return abs(float(actual) - expected) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit result, manuscript, reply, and figure consistency.")
    parser.add_argument(
        "--require-manuscript",
        action="store_true",
        help="Fail if local main.tex and reply.tex are absent; release checkouts omit them.",
    )
    args = parser.parse_args()

    master = json.loads((SUMMARY / "results_master.json").read_text(encoding="utf-8"))
    integrity = json.loads((SUMMARY / "integrity_report.json").read_text(encoding="utf-8"))
    table2 = pd.DataFrame(master["table2_comparisons"])
    table3 = pd.DataFrame(master["table3_attack_comparisons"])
    accuracy = pd.DataFrame(master["method_accuracy"])
    attack = pd.DataFrame(master["method_attack"])
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append((name, bool(passed), evidence))

    add("Raw-result integrity", integrity.get("passed") is True,
        "48,600 metrics; 9,720 tree statistics; 6,480 attacks; zero missing tasks")
    dims = master["dimensions"]
    add("Common seeds", dims["seeds"] == list(range(30)), "Every method uses seeds 0--29")
    add("Method count", len(dims["methods"]) == 12, f"Observed {len(dims['methods'])} methods")

    primary = table2[(table2.method == "mags_fixed") & (table2.baseline == "matched_gini")].iloc[0]
    tuned_cart = table2[(table2.method == "mags_tuned") & (table2.baseline == "cart_tuned_depth")].iloc[0]
    tuned_pruned = table2[(table2.method == "mags_tuned") & (table2.baseline == "cart_pruned")].iloc[0]
    add("Primary mean delta", close(primary.mean_delta, 0.0016967061854447306),
        f"{100 * primary.mean_delta:.2f} pp; 95% CI [{100 * primary.ci95_low:.2f}, {100 * primary.ci95_high:.2f}]")
    add("Primary median and rank test", close(primary.median_delta, 0.0),
        f"median {100 * primary.median_delta:.2f} pp; p={primary.raw_p:.3f}; W/T/L {primary.wins}/{primary.ties}/{primary.losses}")
    add("Tuned MAGS vs tuned CART", close(tuned_cart.median_delta, 0.00028571428571428915),
        f"mean {100 * tuned_cart.mean_delta:.2f} pp; median {100 * tuned_cart.median_delta:.2f} pp")
    add("Tuned MAGS vs pruned CART", close(tuned_pruned.median_delta, -0.002614379084967307),
        f"mean {100 * tuned_pruned.mean_delta:.2f} pp; median {100 * tuned_pruned.median_delta:.2f} pp")

    fixed_attack = table3[(table3.outcome == "attack_success_rate") &
                          (table3.method == "mags_fixed") &
                          (table3.baseline == "matched_gini")].iloc[0]
    add("Matched constrained-attack delta", close(fixed_attack.mean_delta, -0.005240992251867425),
        f"{100 * fixed_attack.mean_delta:.2f} pp; Holm p={fixed_attack.holm_p:.3f}")
    add("Table 2 reporting fields", set(["family", "raw_p", "holm_p", "rank_biserial", "ci95_low", "ci95_high", "wins", "ties", "losses"]).issubset(table2.columns),
        "Family, raw/Holm p, effect, CI, and W/T/L are present")
    add("Table 3 paired fields", set(["raw_p", "holm_p", "rank_biserial", "ci95_low", "ci95_high", "wins", "ties", "losses"]).issubset(table3.columns),
        "Paired delta, raw/Holm p, effect, CI, and W/T/L are present")
    add("Method summaries", len(accuracy) == 12 and len(attack) == 8,
        f"{len(accuracy)} accuracy methods and {len(attack)} attacked tree methods")
    tuned_runtime = accuracy.set_index("method").loc["mags_tuned", "mean_fit_time_sec"]
    add("Archived runtime summary", close(tuned_runtime, 1.3308313859255128),
        f"Tuned MAGS mean fit time {tuned_runtime:.3f} s")

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

    manuscript_paths = [ROOT / "main.tex", ROOT / "reply.tex"]
    if all(path.is_file() for path in manuscript_paths):
        main_text = manuscript_paths[0].read_text(encoding="utf-8")
        reply_text = manuscript_paths[1].read_text(encoding="utf-8")
        combined = main_text + "\n" + reply_text
        required_main = [
            "0.17 percentage points on average",
            "The median is zero",
            "0.19 [$-0.30$, 0.68] & 0.03",
            "$-0.23$ [$-0.72$, 0.25] & $-0.26$",
            "$-0.52$ [$-0.95$, $-0.09$]",
            "1.331 seconds per dataset--seed",
            "seeds 0--29 for every method",
        ]
        required_reply = [
            "0.17 percentage points",
            "median 0.03",
            "median $-0.26$",
            "all 12 methods",
            "same 30 seeds",
            "tuned MAGS requires 1.331 s",
            "Response Figure R1",
            "Response Figure R2",
        ]
        stale = [
            "0.30 percentage points",
            "extended tuning and attacks retain their explicitly stated smaller seed counts",
            "5 seeds for tuned and stronger baselines",
            "10 core seeds and 5 extended seeds",
        ]
        add("Main-manuscript numerical strings", all(value in main_text for value in required_main),
            "Primary, tuned, pruned, attack, and seed statements match the master")
        add("Reply numerical strings", all(value in reply_text for value in required_reply),
            "Primary and corrected medians, 30-seed scope, and R1/R2 labels match")
        add("No stale numerical statements", not any(value in combined for value in stale),
            "No 0.30-pp headline or reduced-seed description remains")
    else:
        add("Local manuscript/reply audit", not args.require_manuscript,
            "Not distributed in the public artifact; use --require-manuscript in the author workspace")

    passed = all(item[1] for item in checks)
    lines = [
        "# Numerical Consistency Audit",
        "",
        "Unique source: `results/summary/results_master.json`, regenerated from the three final raw CSV files.",
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
    raise SystemExit(main())
