"""Evaluate binary robust-tree learners on the existing paired benchmark splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))

from run_experiments import (  # noqa: E402
    exact_continuous_tree_attack,
    load_raw_dataset,
    noisy_continuous_copy,
    preprocess_train_test,
    spec_for,
)


DEFAULT_DATASETS = [
    "adult", "bank_marketing", "breast_cancer", "compas", "credit_approval",
    "credit_g_openml", "diabetes", "german_credit", "heart_statlog",
    "ionosphere", "mushroom", "sonar", "spambase", "tic_tac_toe", "vote",
]
OUTPUT = ROOT / "results" / "raw" / "robust_tree_comparison.csv"


def groot_classifier():
    # groot-trees 0.0.17 imports a NumPy module removed in NumPy 2; only the
    # public np.iterable function is needed by that import.
    if "numpy.lib.function_base" not in sys.modules:
        shim = types.ModuleType("numpy.lib.function_base")
        shim.iterable = np.iterable
        sys.modules[shim.__name__] = shim
    from groot.model import GrootTreeClassifier

    return GrootTreeClassifier


def run_one(dataset: str, seed: int, classifier) -> list[dict]:
    spec = spec_for(dataset)
    X, y, meta = load_raw_dataset(spec, seed)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{dataset} has {len(np.unique(y))} classes; GROOT requires two")
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=stratify
    )
    X_train, X_test, binary_mask, _ = preprocess_train_test(X_train, X_test, meta)
    attack_model = np.where(binary_mask, 0.0, 0.05).tolist()
    rows = []
    for method, chen_heuristic in (("groot", False), ("chen_heuristic", True)):
        model = classifier(
            max_depth=spec.max_depth,
            min_samples_leaf=spec.min_samples_leaf,
            attack_model=attack_model,
            chen_heuristic=chen_heuristic,
            compile=False,
            random_state=seed,
        )
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - started
        clean = accuracy_score(y_test, model.predict(X_test))
        noisy = accuracy_score(
            y_test,
            model.predict(noisy_continuous_copy(X_test, 0.2, seed + 1209, binary_mask)),
        )
        attack = exact_continuous_tree_attack(model, X_test, y_test, binary_mask, eps=0.05)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "clean_accuracy": clean,
            "noisy_accuracy": noisy,
            "fit_time_sec": fit_time,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": X_train.shape[1],
            "n_binary_features": int(binary_mask.sum()),
            **attack,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    classifier = groot_classifier()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        prior = pd.read_csv(args.output)
        complete = set(zip(prior.dataset, prior.seed, prior.method))
    else:
        complete = set()
    for dataset in args.datasets:
        for seed in args.seeds:
            if all((dataset, seed, method) in complete for method in ("groot", "chen_heuristic")):
                continue
            rows = run_one(dataset, seed, classifier)
            # Write one paired task at a time so an interrupted run can resume.
            frame = pd.DataFrame(rows)
            frame.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
            complete.update((dataset, seed, row["method"]) for row in rows)
            print(json.dumps({"dataset": dataset, "seed": seed, "fit_seconds": [round(row["fit_time_sec"], 3) for row in rows]}), flush=True)


if __name__ == "__main__":
    main()
