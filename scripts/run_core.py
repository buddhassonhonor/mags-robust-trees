from __future__ import annotations

import argparse

from _run_stage import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 30-seed implementation-matched core experiment.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    return run_stage("core", "none", "core", args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
