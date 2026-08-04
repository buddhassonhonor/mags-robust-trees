from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_stage(group: str, attacks: str, output_name: str, workers: int) -> int:
    command = [
        sys.executable,
        str(ROOT / "exp" / "run_experiments.py"),
        "--method-group", group,
        "--attacks", attacks,
        "--workers", str(workers),
        "--output", str(ROOT / "exp" / "stages" / output_name),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode
