from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml


OPENML_IDS = [11, 18, 23, 24, 29, 31, 32, 36, 37, 40, 44, 50, 53, 54, 56, 59]
DOWNLOADS = {
    "adult/adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult/adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "car.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    "german/german.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "compas/compas-scores-two-years.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def prepare_bank(root: Path) -> None:
    output = root / "bank" / "bank.csv"
    if output.exists():
        return
    archive = root / "downloads" / "bank.zip"
    download("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip", archive)
    with zipfile.ZipFile(archive) as handle:
        member = next(name for name in handle.namelist() if name.endswith("bank.csv"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)


def prepare_openml(root: Path) -> list[dict]:
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    records = []
    for data_id in OPENML_IDS:
        destination = prepared / f"openml_{data_id}.parquet"
        if not destination.exists():
            print(f"Preparing OpenML dataset {data_id}")
            bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(root / "openml_cache"), parser="auto")
            frame = bunch.data.copy()
            frame["__target__"] = bunch.target
            frame.to_parquet(destination, index=False)
            name = bunch.details.get("name", f"OpenML-{data_id}") if hasattr(bunch, "details") else f"OpenML-{data_id}"
        else:
            name = f"OpenML-{data_id}"
        records.append({"openml_id": data_id, "name": name, "path": destination.relative_to(root).as_posix()})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public datasets for the MAGS experiments.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative, url in DOWNLOADS.items():
        download(url, root / relative)
    prepare_bank(root)
    fetch_covtype(data_home=str(root / "scikit_ml_learn_data"), download_if_missing=True)
    openml_records = prepare_openml(root)
    manifest = {
        "data_root": str(root),
        "raw_data_redistributed": False,
        "sources": {
            "uci": list(DOWNLOADS.values()) + ["https://archive.ics.uci.edu/dataset/222/bank+marketing"],
            "openml_ids": OPENML_IDS,
            "propublica": DOWNLOADS["compas/compas-scores-two-years.csv"],
            "scikit_learn": "https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes",
        },
        "prepared_openml": openml_records,
    }
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared data under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
