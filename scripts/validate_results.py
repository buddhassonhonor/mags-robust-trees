from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"


def unique_count(frame: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    return len(frame), len(frame.drop_duplicates(columns))


def main() -> int:
    metrics = pd.read_csv(RAW / "results.csv")
    stats = pd.read_csv(RAW / "tree_stats.csv")
    attacks = pd.read_csv(RAW / "exact_attacks.csv")
    expected_metrics = len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["methods"]) * len(CONFIG["noise_standard_deviations"])
    expected_stats = len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["methods"])
    expected_attacks = len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["tree_attack_methods"])
    metric_counts = unique_count(metrics, ["dataset", "seed", "method", "noise"])
    stat_counts = unique_count(stats, ["dataset", "seed", "method"])
    attack_counts = unique_count(attacks, ["dataset", "seed", "method"])
    checks = {
        "metrics_rows": metric_counts[0],
        "metrics_unique": metric_counts[1],
        "metrics_expected": expected_metrics,
        "tree_stats_rows": stat_counts[0],
        "tree_stats_unique": stat_counts[1],
        "tree_stats_expected": expected_stats,
        "attack_rows": attack_counts[0],
        "attack_unique": attack_counts[1],
        "attack_expected": expected_attacks,
        "datasets_match": sorted(metrics.dataset.unique()) == sorted(CONFIG["datasets"]),
        "seeds_match": sorted(metrics.seed.unique()) == CONFIG["seeds"],
        "methods_match": sorted(metrics.method.unique()) == sorted(CONFIG["methods"]),
        "attack_methods_match": sorted(attacks.method.unique()) == sorted(CONFIG["tree_attack_methods"]),
        "missing_numeric_metrics": int(metrics[["accuracy", "balanced_accuracy", "macro_f1"]].isna().sum().sum()),
    }
    checks["passed"] = all([
        metric_counts == (expected_metrics, expected_metrics),
        stat_counts == (expected_stats, expected_stats),
        attack_counts == (expected_attacks, expected_attacks),
        checks["datasets_match"], checks["seeds_match"], checks["methods_match"], checks["attack_methods_match"],
        checks["missing_numeric_metrics"] == 0,
    ])
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "integrity_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
