from __future__ import annotations

import argparse

from _run_stage import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run constrained exact opposite-leaf attacks for all single-tree methods on seeds 0--29.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    return run_stage("trees", "all", "attacks", args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
