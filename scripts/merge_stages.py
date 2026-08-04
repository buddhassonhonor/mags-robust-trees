from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "exp" / "stages"
OUT = ROOT / "exp" / "final_validation"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([pd.read_csv(STAGES / stage / "results.csv") for stage in ["core", "baselines"]], ignore_index=True)
    stats = pd.concat([pd.read_csv(STAGES / stage / "tree_stats.csv") for stage in ["core", "baselines"]], ignore_index=True)
    attacks = pd.read_csv(STAGES / "attacks" / "exact_attacks.csv")
    metrics.sort_values(["dataset", "seed", "method", "noise"]).to_csv(OUT / "results.csv", index=False)
    stats.sort_values(["dataset", "seed", "method"]).to_csv(OUT / "tree_stats.csv", index=False)
    attacks.sort_values(["dataset", "seed", "method"]).to_csv(OUT / "exact_attacks.csv", index=False)
    core_meta = json.loads((STAGES / "core" / "metadata.json").read_text(encoding="utf-8"))
    baseline_meta = json.loads((STAGES / "baselines" / "metadata.json").read_text(encoding="utf-8"))
    attack_meta = json.loads((STAGES / "attacks" / "metadata.json").read_text(encoding="utf-8"))
    core_meta["elapsed_sec_by_stage"] = {
        "core": core_meta["elapsed_sec"],
        "baselines": baseline_meta["elapsed_sec"],
        "attacks": attack_meta["elapsed_sec"],
    }
    core_meta["failures"] = core_meta["failures"] + baseline_meta["failures"] + attack_meta["failures"]
    core_meta["method_protocol"] = {
        "all_methods_seeds": 30,
        "exact_attack_tree_methods_seeds": 30,
        "noise": "Gaussian noise on train-standardized non-binary encoded columns only",
        "attack": "Exact under the stated continuous-coordinate and fixed-binary constraints",
    }
    (OUT / "metadata.json").write_text(json.dumps(core_meta, indent=2), encoding="utf-8")
    print(f"Merged stage outputs into {OUT}")
    return 0 if not core_meta["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
