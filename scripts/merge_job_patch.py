from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def replace_rows(base: pd.DataFrame, patch: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    patch_keys = patch[keys].drop_duplicates()
    marked = base.merge(patch_keys.assign(_replace=True), on=keys, how="left")
    kept = marked[marked._replace.isna()].drop(columns="_replace")
    return pd.concat([kept, patch], ignore_index=True).sort_values(keys).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace failed dataset-seed rows with a successful patch run.")
    parser.add_argument("base", type=Path)
    parser.add_argument("patch", type=Path)
    args = parser.parse_args()
    for name, keys in [
        ("results.csv", ["dataset", "seed", "method", "noise"]),
        ("tree_stats.csv", ["dataset", "seed", "method"]),
        ("exact_attacks.csv", ["dataset", "seed", "method"]),
    ]:
        base = pd.read_csv(args.base / name)
        patch = pd.read_csv(args.patch / name)
        replace_rows(base, patch, keys).to_csv(args.base / name, index=False)
    base_meta = json.loads((args.base / "metadata.json").read_text(encoding="utf-8"))
    patch_meta = json.loads((args.patch / "metadata.json").read_text(encoding="utf-8"))
    replacements = {(row["dataset"], row["seed"]) for row in patch_meta["dataset_metadata"]}
    base_meta["dataset_metadata"] = [
        row for row in base_meta["dataset_metadata"] if (row["dataset"], row["seed"]) not in replacements
    ] + patch_meta["dataset_metadata"]
    base_meta["dataset_metadata"] = sorted(base_meta["dataset_metadata"], key=lambda row: (row["dataset"], row["seed"]))
    base_meta["failures"] = []
    base_meta["completed_jobs"] = len(base_meta["dataset_metadata"])
    base_meta["elapsed_sec"] = float(base_meta["elapsed_sec"]) + float(patch_meta["elapsed_sec"])
    (args.base / "metadata.json").write_text(json.dumps(base_meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
