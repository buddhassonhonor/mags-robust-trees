from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = ROOT / "results" / "raw"
    destination = ROOT / "exp" / "final_validation"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["results.csv", "tree_stats.csv", "exact_attacks.csv"]:
        shutil.copy2(source / name, destination / name)
    for name in ["threshold_case.json", "threshold_case_candidates.csv", "threshold_case_points.csv"]:
        archived = ROOT / "results" / "summary" / name
        if archived.exists():
            shutil.copy2(archived, destination / name)
    subprocess.run([sys.executable, str(ROOT / "exp" / "make_figures.py")], check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
