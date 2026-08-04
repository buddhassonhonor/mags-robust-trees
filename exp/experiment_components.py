from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / "src" / "harnesses"))

from margin_aware_splitting_harness import MarginAwareTree, generate_data  # noqa: E402


DATA_ROOT = Path(os.environ.get("MAGS_DATA_ROOT", PACKAGE_DIR / "data")).resolve()
OUT_DIR = PACKAGE_DIR / "exp" / "eswa_main"
RAW_CSV = OUT_DIR / "results.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
DELTAS_CSV = OUT_DIR / "paired_deltas.csv"
STAT_CSV = OUT_DIR / "stat_tests.csv"
META_JSON = OUT_DIR / "metadata.json"
REPORT_MD = OUT_DIR / "report.md"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    target: str | None = None
    path: str | None = None
    openml_id: int | None = None
    max_samples: int | None = 3000
    max_depth: int = 5
    min_samples_leaf: int = 4
    max_thresholds: int = 32


DATASETS = [
    DatasetSpec("synthetic_xor", "synthetic", max_samples=1800, max_depth=3, min_samples_leaf=1, max_thresholds=64),
    DatasetSpec("breast_cancer", "sklearn", max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=40),
    DatasetSpec("wine", "sklearn", max_samples=None, max_depth=4, min_samples_leaf=3, max_thresholds=32),
    DatasetSpec("digits", "sklearn", max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("iris", "sklearn", max_samples=None, max_depth=3, min_samples_leaf=2, max_thresholds=32),
    DatasetSpec("covertype_3000", "sklearn_covtype", max_samples=3000, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("adult", "uci_adult", max_samples=3000, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("bank_marketing", "csv_semicolon", path="bank/bank.csv", target="y", max_samples=None, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("car_evaluation", "uci_car", max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("german_credit", "uci_german", max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("compas", "compas", max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("balance_scale", "openml", openml_id=11, max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("mfeat_morphological", "openml", openml_id=18, max_samples=3000, max_depth=5, min_samples_leaf=4, max_thresholds=24),
    DatasetSpec("cmc", "openml", openml_id=23, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("mushroom", "openml", openml_id=24, max_samples=3000, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("credit_approval", "openml", openml_id=29, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("credit_g_openml", "openml", openml_id=31, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("pendigits", "openml", openml_id=32, max_samples=3000, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("segment", "openml", openml_id=36, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("diabetes", "openml", openml_id=37, max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("sonar", "openml", openml_id=40, max_samples=None, max_depth=4, min_samples_leaf=3, max_thresholds=32),
    DatasetSpec("spambase", "local_openml_parquet", path="openml/org/openml/www/datasets/44/dataset_44.pq", target="class", openml_id=44, max_samples=3000, max_depth=6, min_samples_leaf=8, max_thresholds=24),
    DatasetSpec("tic_tac_toe", "openml", openml_id=50, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("heart_statlog", "openml", openml_id=53, max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("vehicle", "openml", openml_id=54, max_samples=None, max_depth=5, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("vote", "openml", openml_id=56, max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=32),
    DatasetSpec("ionosphere", "local_openml_parquet", path="openml/org/openml/www/datasets/59/dataset_59.pq", target="class", openml_id=59, max_samples=None, max_depth=4, min_samples_leaf=4, max_thresholds=32),
]

SEEDS = [0, 1, 2, 3, 4]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
UNIFORM_EPS = [0.05, 0.10, 0.20]
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def set_thread_limits() -> None:
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, "1")


def label_encode(y: pd.Series | np.ndarray) -> np.ndarray:
    # Canonicalize labels independently of provider-specific categorical order.
    return pd.Series(y).astype(str).astype("category").cat.codes.to_numpy(dtype=int)


def preprocess_frame(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, dict]:
    df = df.copy()
    y = label_encode(df.pop(target))
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) == "category":
            df[col] = df[col].astype("category")
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or str(df[c].dtype) == "category"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    transformer = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        sparse_threshold=0.0,
    )
    X = transformer.fit_transform(df)
    meta = {
        "n_instances": int(len(y)),
        "n_raw_features": int(df.shape[1]),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "n_encoded_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }
    return np.asarray(X, dtype=float), y, meta


def load_dataset(spec: DatasetSpec, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if spec.source == "synthetic":
        X, y = generate_data(seed, n_samples=spec.max_samples or 1800)
        return X.astype(float), y.astype(int), {
            "n_instances": int(X.shape[0]),
            "n_raw_features": int(X.shape[1]),
            "n_numeric_features": int(X.shape[1]),
            "n_categorical_features": 0,
            "n_encoded_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
    if spec.source == "sklearn":
        loader = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits, "iris": load_iris}[spec.name]
        data = loader()
        X = data.data.astype(float)
        y = data.target.astype(int)
        return X, y, {
            "n_instances": int(X.shape[0]),
            "n_raw_features": int(X.shape[1]),
            "n_numeric_features": int(X.shape[1]),
            "n_categorical_features": 0,
            "n_encoded_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
    if spec.source == "sklearn_covtype":
        data = fetch_covtype(data_home=str(DATA_ROOT / "scikit_ml_learn_data"))
        X = data.data.astype(float)
        y = data.target.astype(int) - 1
        return sample_rows(X, y, spec.max_samples, seed), y_sampled(y, spec.max_samples, seed), {
            "n_instances": int(min(spec.max_samples or len(y), len(y))),
            "n_raw_features": int(X.shape[1]),
            "n_numeric_features": int(X.shape[1]),
            "n_categorical_features": 0,
            "n_encoded_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
    if spec.source == "uci_adult":
        cols = [
            "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income",
        ]
        train = pd.read_csv(DATA_ROOT / "adult" / "adult.data", names=cols, na_values=" ?", skipinitialspace=True)
        test = pd.read_csv(DATA_ROOT / "adult" / "adult.test", names=cols, na_values=" ?", skipinitialspace=True, comment="|")
        df = pd.concat([train, test], ignore_index=True)
        df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        X, y, meta = preprocess_frame(df, "income")
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "csv_semicolon":
        df = pd.read_csv(DATA_ROOT / (spec.path or ""), sep=";")
        X, y, meta = preprocess_frame(df, spec.target or "target")
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "uci_car":
        cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        df = pd.read_csv(DATA_ROOT / "car.data", names=cols)
        X, y, meta = preprocess_frame(df, "class")
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "uci_german":
        cols = [f"x{i}" for i in range(20)] + ["target"]
        df = pd.read_csv(DATA_ROOT / "german" / "german.data", sep=r"\s+", names=cols)
        X, y, meta = preprocess_frame(df, "target")
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "compas":
        path = DATA_ROOT / "compas" / "compas-scores-two-years.csv"
        df = pd.read_csv(path)
        keep = [
            "sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid",
        ]
        df = df[keep].dropna()
        X, y, meta = preprocess_frame(df, "two_year_recid")
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "local_openml_parquet":
        df = pd.read_parquet(DATA_ROOT / (spec.path or ""))
        X, y, meta = preprocess_frame(df, spec.target or df.columns[-1])
        meta["openml_id"] = spec.openml_id
        return sample_pair(X, y, spec.max_samples, seed, meta)
    if spec.source == "openml":
        import openml

        openml.config.set_root_cache_directory(str(DATA_ROOT / "openml_cache"))
        ds = openml.datasets.get_dataset(spec.openml_id, download_data=True, download_qualities=True, download_features_meta_data=True)
        X_df, y_ser, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
        df = X_df.copy()
        df["__target__"] = y_ser
        X, y, meta = preprocess_frame(df, "__target__")
        meta["openml_id"] = spec.openml_id
        return sample_pair(X, y, spec.max_samples, seed, meta)
    raise ValueError(f"unknown dataset source: {spec.source}")


def y_sampled(y: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(y) <= max_samples:
        return y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return y[idx]


def sample_rows(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(y) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return X[idx]


def sample_pair(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int, meta: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    meta = dict(meta)
    meta["n_instances_used"] = int(len(y))
    return X, y, meta


def noisy_copy(X: np.ndarray, noise: float, seed: int, mode: str = "gaussian") -> np.ndarray:
    if noise == 0:
        return X
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        return X + rng.uniform(-noise, noise, size=X.shape)
    return X + rng.normal(0.0, noise, size=X.shape)


def tree_stats(node: dict, depth: int = 0) -> tuple[int, int, int, list[float]]:
    if node.get("leaf", False):
        return 1, 1, depth, []
    left_nodes, left_leaves, left_depth, left_margins = tree_stats(node["left"], depth + 1)
    right_nodes, right_leaves, right_depth, right_margins = tree_stats(node["right"], depth + 1)
    margins = [float(node.get("norm_margin", 0.0))] + left_margins + right_margins
    return 1 + left_nodes + right_nodes, left_leaves + right_leaves, max(left_depth, right_depth), margins


def path_min_margin_one(model: MarginAwareTree, x: np.ndarray) -> float:
    node = model.tree
    margins = []
    while not node.get("leaf", False):
        feat = node["feature_idx"]
        threshold = node["threshold"]
        margins.append(abs(float(x[feat]) - float(threshold)))
        node = node["left"] if x[feat] <= threshold else node["right"]
    return min(margins) if margins else float("inf")


def threshold_crossing_eval(model: MarginAwareTree, X: np.ndarray, y: np.ndarray, eps: float = 0.05) -> tuple[float, float]:
    pred = model.predict(X)
    attacked = X.copy()
    for i, x in enumerate(X):
        node = model.tree
        best_feat = None
        best_dist = float("inf")
        best_threshold = 0.0
        while not node.get("leaf", False):
            feat = node["feature_idx"]
            threshold = float(node["threshold"])
            dist = abs(float(x[feat]) - threshold)
            if dist < best_dist:
                best_feat = feat
                best_dist = dist
                best_threshold = threshold
            node = node["left"] if x[feat] <= threshold else node["right"]
        if best_feat is not None and best_dist <= eps:
            direction = 1.0 if x[best_feat] <= best_threshold else -1.0
            attacked[i, best_feat] = best_threshold + direction * (eps + 1e-6)
    attacked_pred = model.predict(attacked)
    return float(np.mean(pred != attacked_pred)), float(accuracy_score(y, attacked_pred))


def fit_margin(X_train: np.ndarray, y_train: np.ndarray, spec: DatasetSpec, method: str, alpha: float, score_mode: str) -> tuple[MarginAwareTree, dict]:
    started = time.perf_counter()
    model = MarginAwareTree(
        max_depth=spec.max_depth,
        alpha=alpha,
        max_thresholds=spec.max_thresholds,
        min_samples_leaf=spec.min_samples_leaf,
        score_mode=score_mode,
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    n_nodes, n_leaves, actual_depth, margins = tree_stats(model.tree)
    return model, {
        "method": method,
        "method_family": "margin_tree",
        "alpha": alpha,
        "score_mode": score_mode,
        "fit_time_sec": fit_time,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "actual_depth": actual_depth,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "min_node_margin": float(np.min(margins)) if margins else 0.0,
    }


def fit_tuned_margin(X_train: np.ndarray, y_train: np.ndarray, spec: DatasetSpec, seed: int) -> tuple[MarginAwareTree, dict]:
    stratify = y_train if min(np.bincount(y_train)) >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed + 101, stratify=stratify)
    best = None
    started = time.perf_counter()
    for depth in [max(2, spec.max_depth - 2), max(2, spec.max_depth - 1), spec.max_depth, spec.max_depth + 1, min(10, spec.max_depth + 3)]:
        for alpha in ALPHAS:
            model = MarginAwareTree(
                max_depth=depth,
                alpha=alpha,
                max_thresholds=spec.max_thresholds,
                min_samples_leaf=spec.min_samples_leaf,
                score_mode="gain_gated",
            )
            model.fit(X_fit, y_fit)
            X_val_noisy = noisy_copy(X_val, 0.10, seed + 8080, "gaussian")
            score = balanced_accuracy_score(y_val, model.predict(X_val_noisy))
            nodes, leaves, actual_depth, margins = tree_stats(model.tree)
            # Prefer simpler trees when validation scores tie.
            tie_break = -leaves
            candidate = (score, tie_break, alpha, depth)
            if best is None or candidate > best:
                best = candidate
    _, _, alpha, depth = best
    model = MarginAwareTree(
        max_depth=depth,
        alpha=alpha,
        max_thresholds=spec.max_thresholds,
        min_samples_leaf=spec.min_samples_leaf,
        score_mode="gain_gated",
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    n_nodes, n_leaves, actual_depth, margins = tree_stats(model.tree)
    return model, {
        "method": "mags_tuned",
        "method_family": "margin_tree",
        "alpha": alpha,
        "score_mode": f"gain_gated;depth={depth}",
        "fit_time_sec": fit_time,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "actual_depth": actual_depth,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "min_node_margin": float(np.min(margins)) if margins else 0.0,
    }


def sklearn_tree_stats(model: DecisionTreeClassifier) -> dict:
    return {
        "n_nodes": int(model.tree_.node_count),
        "n_leaves": int(model.get_n_leaves()),
        "actual_depth": int(model.get_depth()),
        "avg_margin": "",
        "min_node_margin": "",
    }


def fit_sklearn_model(X_train: np.ndarray, y_train: np.ndarray, spec: DatasetSpec, method: str, seed: int):
    if method == "cart_gini":
        model = DecisionTreeClassifier(max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf, criterion="gini", random_state=seed)
    elif method == "cart_entropy":
        model = DecisionTreeClassifier(max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf, criterion="entropy", random_state=seed)
    elif method == "cart_tuned_depth":
        return fit_depth_tuned_cart(X_train, y_train, spec, seed)
    elif method == "cart_pruned":
        return fit_pruned_cart(X_train, y_train, spec, seed)
    elif method == "noise_augmented_cart":
        rng = np.random.default_rng(seed + 9100)
        X_aug = np.vstack([X_train, X_train + rng.normal(0.0, 0.10, size=X_train.shape)])
        y_aug = np.concatenate([y_train, y_train])
        model = DecisionTreeClassifier(max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf, criterion="gini", random_state=seed)
        started = time.perf_counter()
        model.fit(X_aug, y_aug)
        fit_time = time.perf_counter() - started
        meta = {"method": method, "method_family": "sklearn", "alpha": "", "score_mode": "", "fit_time_sec": fit_time, **sklearn_tree_stats(model)}
        return model, meta
    elif method == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf, random_state=seed, n_jobs=1)
    elif method == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=100, max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf, random_state=seed, n_jobs=1)
    else:
        raise ValueError(method)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    if isinstance(model, DecisionTreeClassifier):
        stats = sklearn_tree_stats(model)
    else:
        stats = {"n_nodes": "", "n_leaves": "", "actual_depth": spec.max_depth, "avg_margin": "", "min_node_margin": ""}
    meta = {"method": method, "method_family": "sklearn", "alpha": "", "score_mode": "", "fit_time_sec": fit_time, **stats}
    return model, meta


def fit_depth_tuned_cart(X_train: np.ndarray, y_train: np.ndarray, spec: DatasetSpec, seed: int):
    stratify = y_train if min(np.bincount(y_train)) >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed, stratify=stratify)
    best = None
    started = time.perf_counter()
    for depth in [2, 3, 4, 5, 6, 8, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=spec.min_samples_leaf, criterion="gini", random_state=seed)
        model.fit(X_fit, y_fit)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        if best is None or score > best[0]:
            best = (score, depth)
    model = DecisionTreeClassifier(max_depth=best[1], min_samples_leaf=spec.min_samples_leaf, criterion="gini", random_state=seed)
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    return model, {"method": "cart_tuned_depth", "method_family": "sklearn", "alpha": "", "score_mode": f"depth={best[1]}", "fit_time_sec": fit_time, **sklearn_tree_stats(model)}


def fit_pruned_cart(X_train: np.ndarray, y_train: np.ndarray, spec: DatasetSpec, seed: int):
    stratify = y_train if min(np.bincount(y_train)) >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed + 7, stratify=stratify)
    base = DecisionTreeClassifier(max_depth=None, min_samples_leaf=spec.min_samples_leaf, criterion="gini", random_state=seed)
    path = base.cost_complexity_pruning_path(X_fit, y_fit)
    ccp_alphas = np.unique(np.maximum(path.ccp_alphas, 0.0))
    if len(ccp_alphas) > 12:
        ccp_alphas = np.quantile(ccp_alphas, np.linspace(0, 0.95, 12))
    best = None
    started = time.perf_counter()
    for ccp_alpha in ccp_alphas:
        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=spec.min_samples_leaf, criterion="gini", ccp_alpha=float(ccp_alpha), random_state=seed)
        model.fit(X_fit, y_fit)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        if best is None or score > best[0]:
            best = (score, float(ccp_alpha))
    model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=spec.min_samples_leaf, criterion="gini", ccp_alpha=best[1], random_state=seed)
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    return model, {"method": "cart_pruned", "method_family": "sklearn", "alpha": "", "score_mode": f"ccp_alpha={best[1]:.6g}", "fit_time_sec": fit_time, **sklearn_tree_stats(model)}


def eval_rows(model, meta: dict, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> list[dict]:
    rows = []
    for noise in NOISE_LEVELS:
        X_eval = noisy_copy(X_test, noise, seed + int(noise * 1000) + 17, "gaussian")
        started = time.perf_counter()
        pred = model.predict(X_eval)
        predict_time = time.perf_counter() - started
        rows.append({
            **meta,
            "perturbation": "gaussian",
            "noise": noise,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "predict_time_sec": predict_time,
            "flip_rate": "",
            "threshold_attack_accuracy": "",
        })
    for eps in UNIFORM_EPS:
        X_eval = noisy_copy(X_test, eps, seed + int(eps * 1000) + 29, "uniform")
        started = time.perf_counter()
        pred = model.predict(X_eval)
        predict_time = time.perf_counter() - started
        rows.append({
            **meta,
            "perturbation": "uniform_linf",
            "noise": eps,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "predict_time_sec": predict_time,
            "flip_rate": "",
            "threshold_attack_accuracy": "",
        })
    if isinstance(model, MarginAwareTree):
        flip_rate, adv_acc = threshold_crossing_eval(model, X_test, y_test, eps=0.05)
        rows.append({
            **meta,
            "perturbation": "threshold_crossing",
            "noise": 0.05,
            "accuracy": "",
            "balanced_accuracy": "",
            "macro_f1": "",
            "predict_time_sec": "",
            "flip_rate": flip_rate,
            "threshold_attack_accuracy": adv_acc,
        })
    return rows


def run_one(dataset_name: str, seed: int) -> tuple[list[dict], dict | None, str | None]:
    set_thread_limits()
    spec = next(s for s in DATASETS if s.name == dataset_name)
    try:
        X, y, data_meta = load_dataset(spec, seed)
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
            return [], None, f"{dataset_name} seed={seed}: too few classes after loading"
        stratify = y if min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=stratify)
        rows: list[dict] = []
        dataset_meta = {
            "dataset": dataset_name,
            "seed": seed,
            "source": spec.source,
            "max_depth": spec.max_depth,
            "min_samples_leaf": spec.min_samples_leaf,
            "max_thresholds": spec.max_thresholds,
            **data_meta,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
        }

        margin_methods = [
            ("mags_a0", 0.0, "gain_gated"),
            ("mags_a01", 0.1, "gain_gated"),
            ("mags_a03", 0.3, "gain_gated"),
            ("mags_a05", 0.5, "gain_gated"),
            ("mags_a07", 0.7, "gain_gated"),
            ("mags_a10", 1.0, "gain_gated"),
            ("legacy_convex_a05", 0.5, "convex"),
            ("margin_only", 1.0, "margin_only"),
            ("unnormalized_margin_a05", 0.5, "unnormalized_margin"),
        ]
        sklearn_methods = [
            "cart_gini",
            "cart_entropy",
            "cart_tuned_depth",
            "cart_pruned",
            "noise_augmented_cart",
            "random_forest",
            "extra_trees",
        ]
        for method, alpha, score_mode in margin_methods:
            model, meta = fit_margin(X_train, y_train, spec, method, alpha, score_mode)
            rows.extend({**dataset_meta, **row} for row in eval_rows(model, meta, X_test, y_test, seed))
        model, meta = fit_tuned_margin(X_train, y_train, spec, seed)
        rows.extend({**dataset_meta, **row} for row in eval_rows(model, meta, X_test, y_test, seed))
        for method in sklearn_methods:
            model, meta = fit_sklearn_model(X_train, y_train, spec, method, seed)
            rows.extend({**dataset_meta, **row} for row in eval_rows(model, meta, X_test, y_test, seed))
        return rows, dataset_meta, None
    except Exception as exc:
        return [], None, f"{dataset_name} seed={seed}: {type(exc).__name__}: {exc}"


def summarize_values(values: list[float]) -> dict:
    vals = [float(v) for v in values if not pd.isna(v)]
    if not vals:
        return {"n": 0, "mean": "", "std": "", "ci95_low": "", "ci95_high": "", "median": "", "iqr": "", "min": "", "max": ""}
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    half_width = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    return {
        "n": len(vals),
        "mean": m,
        "std": sd,
        "ci95_low": m - half_width,
        "ci95_high": m + half_width,
        "median": float(np.median(vals)),
        "iqr": float(q75 - q25),
        "min": min(vals),
        "max": max(vals),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [1.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def build_aggregates(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    df = pd.DataFrame(rows)
    metric_df = df[df["accuracy"] != ""].copy()
    metric_df["accuracy"] = metric_df["accuracy"].astype(float)
    metric_df["balanced_accuracy"] = metric_df["balanced_accuracy"].astype(float)
    metric_df["macro_f1"] = metric_df["macro_f1"].astype(float)

    summary_rows = []
    for keys, group in metric_df.groupby(["dataset", "method", "perturbation", "noise"], dropna=False):
        dataset, method, perturbation, noise = keys
        row = {"dataset": dataset, "method": method, "perturbation": perturbation, "noise": noise}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "fit_time_sec", "predict_time_sec"]:
            stats = summarize_values(group[metric].astype(float).tolist())
            row.update({f"{metric}_{k}": v for k, v in stats.items()})
        summary_rows.append(row)

    delta_rows = []
    comparisons = ["mags_a05", "cart_gini", "cart_tuned_depth", "cart_pruned", "noise_augmented_cart", "legacy_convex_a05", "margin_only"]
    ours = metric_df[metric_df["method"] == "mags_tuned"]
    for baseline in comparisons:
        base = metric_df[metric_df["method"] == baseline]
        merged = ours.merge(base, on=["dataset", "seed", "perturbation", "noise"], suffixes=("_ours", "_base"))
        for _, row in merged.iterrows():
            delta_rows.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "perturbation": row["perturbation"],
                "noise": row["noise"],
                "baseline": baseline,
                "accuracy_delta": float(row["accuracy_ours"] - row["accuracy_base"]),
                "balanced_accuracy_delta": float(row["balanced_accuracy_ours"] - row["balanced_accuracy_base"]),
                "macro_f1_delta": float(row["macro_f1_ours"] - row["macro_f1_base"]),
            })

    stat_rows = []
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        for keys, group in delta_df.groupby(["baseline", "perturbation", "noise"]):
            baseline, perturbation, noise = keys
            vals = group.groupby("dataset")["accuracy_delta"].mean()
            raw_p = 1.0
            if len(vals) >= 2 and np.any(np.asarray(vals) != 0):
                try:
                    raw_p = float(wilcoxon(vals, zero_method="wilcox", alternative="greater").pvalue)
                except Exception:
                    raw_p = 1.0
            stat_rows.append({
                "baseline": baseline,
                "perturbation": perturbation,
                "noise": noise,
                "n_datasets": int(len(vals)),
                "mean_dataset_delta": float(vals.mean()) if len(vals) else "",
                "median_dataset_delta": float(vals.median()) if len(vals) else "",
                "wins": int((vals > 0).sum()),
                "ties": int((vals == 0).sum()),
                "losses": int((vals < 0).sum()),
                "wilcoxon_p": raw_p,
            })
        pvals = [r["wilcoxon_p"] for r in stat_rows]
        adj = holm_adjust(pvals)
        for row, p in zip(stat_rows, adj):
            row["holm_p"] = p
    return summary_rows, delta_rows, stat_rows


def make_report(summary_rows: list[dict], stat_rows: list[dict], failures: list[str], dataset_meta: list[dict]) -> str:
    df = pd.DataFrame(summary_rows)
    stat = pd.DataFrame(stat_rows)
    lines = [
        "# ESWA Main Experiment Report",
        "",
        f"- Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Datasets attempted: {len(DATASETS)}",
        f"- Dataset-seed failures: {len(failures)}",
        f"- Seeds: {SEEDS}",
        f"- Noise levels: {NOISE_LEVELS}",
        f"- Uniform L-infinity eps: {UNIFORM_EPS}",
        "",
        "## Dataset Coverage",
        "",
        f"- Completed dataset metadata rows: {len(dataset_meta)}",
        f"- Unique completed datasets: {len(set(m['dataset'] for m in dataset_meta)) if dataset_meta else 0}",
        "",
        "## Main Statistical Tests",
        "",
    ]
    if not stat.empty:
        focus = stat[(stat["baseline"] == "cart_gini") & (stat["perturbation"] == "gaussian") & (stat["noise"].astype(float) == 0.2)]
        for _, row in focus.iterrows():
            lines.append(
        f"- Tuned MAGS vs CART Gini, Gaussian noise=0.2: mean dataset delta={row['mean_dataset_delta']:.4f}, "
                f"wins/ties/losses={row['wins']}/{row['ties']}/{row['losses']}, Wilcoxon p={row['wilcoxon_p']:.4g}, Holm p={row['holm_p']:.4g}."
            )
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {f}" for f in failures] or ["- None."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(s.name for s in DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    jobs = [(d, s) for d in dataset_names for s in seeds]
    rows: list[dict] = []
    metas: list[dict] = []
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, d, s): (d, s) for d, s in jobs}
        for future in as_completed(futures):
            d, s = futures[future]
            sub_rows, meta, failure = future.result()
            if failure:
                failures.append(failure)
                print(f"[failed] {failure}", flush=True)
            else:
                print(f"[done] {d} seed={s} rows={len(sub_rows)}", flush=True)
            rows.extend(sub_rows)
            if meta is not None:
                metas.append(meta)
    write_csv(RAW_CSV, rows)
    summary_rows, delta_rows, stat_rows = build_aggregates(rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DELTAS_CSV, delta_rows)
    write_csv(STAT_CSV, stat_rows)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": time.perf_counter() - started,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "datasets": [s.__dict__ for s in DATASETS if s.name in dataset_names],
        "seeds": seeds,
        "failures": failures,
        "dataset_metadata": metas,
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(make_report(summary_rows, stat_rows, failures, metas), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "failures": len(failures), "elapsed_sec": metadata["elapsed_sec"]}, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
