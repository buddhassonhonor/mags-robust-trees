from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr, t, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_covtype, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "src" / "harnesses"))

from experiment_components import (  # noqa: E402
    DATASETS,
    DATA_ROOT,
    DatasetSpec,
    fit_depth_tuned_cart,
    fit_pruned_cart,
    fit_sklearn_model,
    fit_tuned_margin,
)
from margin_aware_splitting_harness import MarginAwareTree, generate_data  # noqa: E402


OUT = ROOT / "exp" / "final_validation"
COMPLETE_DATASETS = [
    "adult", "balance_scale", "bank_marketing", "breast_cancer",
    "car_evaluation", "cmc", "compas", "covertype_3000",
    "credit_approval", "credit_g_openml", "diabetes", "digits",
    "german_credit", "heart_statlog", "iris", "mfeat_morphological",
    "mushroom", "pendigits", "segment", "sonar", "synthetic_xor",
    "spambase", "ionosphere", "tic_tac_toe", "vehicle", "vote", "wine",
]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
CORE_METHODS = ["matched_gini", "mags_fixed", "mags_binary_excluded", "cart_fixed"]
EXTENDED_METHODS = [
    "mags_tuned", "cart_tuned_depth", "cart_pruned", "noise_augmented_cart",
    "random_forest", "extra_trees", "linear_svm", "rbf_svm",
]
TREE_METHODS = [
    "matched_gini", "mags_fixed", "mags_binary_excluded", "cart_fixed",
    "mags_tuned", "cart_tuned_depth", "cart_pruned", "noise_augmented_cart",
]


def set_thread_limits() -> None:
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(name, "1")


def spec_for(name: str) -> DatasetSpec:
    return next(s for s in DATASETS if s.name == name)


def label_encode(y: pd.Series | np.ndarray) -> np.ndarray:
    # Canonicalize labels independently of provider-specific categorical order.
    return pd.Series(y).astype(str).astype("category").cat.codes.to_numpy(dtype=int)


def sample_raw(X: pd.DataFrame, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X.reset_index(drop=True), np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return X.iloc[idx].reset_index(drop=True), np.asarray(y, dtype=int)[idx]


def load_raw_dataset(spec: DatasetSpec, seed: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    if spec.source == "synthetic":
        X, y = generate_data(seed, n_samples=spec.max_samples or 1800)
        frame = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    elif spec.source == "sklearn":
        loader = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits, "iris": load_iris}[spec.name]
        data = loader()
        frame = pd.DataFrame(data.data, columns=getattr(data, "feature_names", None))
        y = np.asarray(data.target, dtype=int)
    elif spec.source == "sklearn_covtype":
        data = fetch_covtype(data_home=str(DATA_ROOT / "scikit_ml_learn_data"))
        frame = pd.DataFrame(data.data)
        y = np.asarray(data.target, dtype=int) - 1
    elif spec.source == "uci_adult":
        cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
        train = pd.read_csv(DATA_ROOT / "adult" / "adult.data", names=cols, na_values=" ?", skipinitialspace=True)
        test = pd.read_csv(DATA_ROOT / "adult" / "adult.test", names=cols, na_values=" ?", skipinitialspace=True, comment="|")
        data = pd.concat([train, test], ignore_index=True)
        data["income"] = data["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        y = label_encode(data.pop("income"))
        frame = data
    elif spec.source == "csv_semicolon":
        data = pd.read_csv(DATA_ROOT / (spec.path or ""), sep=";")
        y = label_encode(data.pop(spec.target or "target"))
        frame = data
    elif spec.source == "uci_car":
        cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        data = pd.read_csv(DATA_ROOT / "car.data", names=cols)
        y = label_encode(data.pop("class"))
        frame = data
    elif spec.source == "uci_german":
        cols = [f"x{i}" for i in range(20)] + ["target"]
        data = pd.read_csv(DATA_ROOT / "german" / "german.data", sep=r"\s+", names=cols)
        y = label_encode(data.pop("target"))
        frame = data
    elif spec.source == "compas":
        data = pd.read_csv(DATA_ROOT / "compas" / "compas-scores-two-years.csv")
        keep = ["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]
        data = data[keep].dropna().reset_index(drop=True)
        y = label_encode(data.pop("two_year_recid"))
        frame = data
    elif spec.source in {"openml", "local_openml_parquet"}:
        canonical = DATA_ROOT / "prepared" / f"openml_{spec.openml_id}.parquet"
        if canonical.exists():
            path = canonical
        elif spec.source == "local_openml_parquet":
            path = DATA_ROOT / (spec.path or "")
        else:
            path = DATA_ROOT / "openml" / "org" / "openml" / "www" / "datasets" / str(spec.openml_id) / f"dataset_{spec.openml_id}.pq"
        data = pd.read_parquet(path)
        target = spec.target if spec.target in data.columns else data.columns[-1]
        y = label_encode(data.pop(target))
        frame = data
    else:
        raise ValueError(f"Unsupported source {spec.source}")
    n_original = len(y)
    frame, y = sample_raw(frame, y, spec.max_samples, seed)
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or str(frame[col].dtype) == "category":
            frame[col] = frame[col].astype("object")
    cat_cols = [c for c in frame.columns if pd.api.types.is_object_dtype(frame[c]) or str(frame[c].dtype) == "category"]
    return frame, y, {
        "n_instances": int(n_original),
        "n_instances_used": int(len(y)),
        "n_raw_features": int(frame.shape[1]),
        "n_numeric_features": int(frame.shape[1] - len(cat_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "n_classes": int(len(np.unique(y))),
        **({"openml_id": spec.openml_id} if spec.openml_id is not None else {}),
    }


def preprocess_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, meta: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cat_cols = [c for c in X_train.columns if pd.api.types.is_object_dtype(X_train[c]) or str(X_train[c].dtype) == "category"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    transformer = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], sparse_threshold=0.0)
    train = np.asarray(transformer.fit_transform(X_train), dtype=float)
    test = np.asarray(transformer.transform(X_test), dtype=float)
    binary = np.zeros(train.shape[1], dtype=bool)
    binary[len(num_cols):] = True
    for j in range(len(num_cols)):
        if len(np.unique(train[:, j])) <= 2:
            binary[j] = True
    updated = dict(meta)
    updated["n_encoded_features"] = int(train.shape[1])
    return train, test, binary, updated


def infer_binary_mask(X: np.ndarray, meta: dict) -> np.ndarray:
    """Identify one-hot and intrinsically binary columns without using labels."""
    mask = np.zeros(X.shape[1], dtype=bool)
    n_numeric = int(meta.get("n_numeric_features", X.shape[1]))
    if n_numeric < X.shape[1]:
        mask[n_numeric:] = True
    for j in range(min(n_numeric, X.shape[1])):
        vals = np.unique(X[:, j])
        if len(vals) <= 2:
            mask[j] = True
    return mask


def scale_continuous_train_only(
    X_train: np.ndarray, X_test: np.ndarray, binary_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float).copy()
    X_test = np.asarray(X_test, dtype=float).copy()
    continuous = ~binary_mask
    if np.any(continuous):
        scaler = StandardScaler()
        X_train[:, continuous] = scaler.fit_transform(X_train[:, continuous])
        X_test[:, continuous] = scaler.transform(X_test[:, continuous])
    return X_train, X_test


def noisy_continuous_copy(X: np.ndarray, sigma: float, seed: int, binary_mask: np.ndarray) -> np.ndarray:
    if sigma == 0:
        return X.copy()
    out = X.copy()
    continuous = ~binary_mask
    rng = np.random.default_rng(seed)
    out[:, continuous] += rng.normal(0.0, sigma, size=(len(X), int(np.sum(continuous))))
    return out


def custom_stats(model: MarginAwareTree) -> dict:
    nodes = leaves = max_depth = binary_nodes = 0
    margins: list[float] = []

    def walk(node: dict, depth: int) -> None:
        nonlocal nodes, leaves, max_depth, binary_nodes
        nodes += 1
        max_depth = max(max_depth, depth)
        if node.get("leaf", False):
            leaves += 1
            return
        margins.append(float(node.get("norm_margin", 0.0)))
        binary_nodes += int(bool(node.get("is_binary", False)))
        walk(node["left"], depth + 1)
        walk(node["right"], depth + 1)

    walk(model.tree, 0)
    split_nodes = nodes - leaves
    return {
        "n_nodes": nodes,
        "n_leaves": leaves,
        "depth": max_depth,
        "mean_node_margin": float(np.mean(margins)) if margins else 0.0,
        "binary_split_fraction": binary_nodes / split_nodes if split_nodes else 0.0,
    }


def sklearn_stats(model: DecisionTreeClassifier) -> dict:
    tree = model.tree_
    split = tree.feature >= 0
    return {
        "n_nodes": int(tree.node_count),
        "n_leaves": int(model.get_n_leaves()),
        "depth": int(model.get_depth()),
        "mean_node_margin": np.nan,
        "binary_split_fraction": np.nan,
    }


def fit_custom(
    X: np.ndarray,
    y: np.ndarray,
    spec: DatasetSpec,
    method: str,
    alpha: float,
    binary_mask: np.ndarray,
    binary_policy: str = "standard",
) -> tuple[MarginAwareTree, dict]:
    started = time.perf_counter()
    model = MarginAwareTree(
        max_depth=spec.max_depth,
        alpha=alpha,
        max_thresholds=spec.max_thresholds,
        min_samples_leaf=spec.min_samples_leaf,
        score_mode="gain_gated",
        binary_features=binary_mask,
        binary_margin_policy=binary_policy,
    )
    model.fit(X, y)
    stats = custom_stats(model)
    stats.update(method=method, fit_time_sec=time.perf_counter() - started)
    return model, stats


def enumerate_custom_leaves(model: MarginAwareTree, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, np.ndarray, int]] = []

    def walk(node: dict, low: np.ndarray, high: np.ndarray) -> None:
        if node.get("leaf", False):
            rows.append((low.copy(), high.copy(), int(node["class"])))
            return
        j = int(node["feature_idx"])
        v = float(node["threshold"])
        left_high = high.copy()
        left_high[j] = min(left_high[j], v)
        walk(node["left"], low, left_high)
        right_low = low.copy()
        right_low[j] = max(right_low[j], v)
        walk(node["right"], right_low, high)

    walk(model.tree, np.full(n_features, -np.inf), np.full(n_features, np.inf))
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def enumerate_sklearn_leaves(model: DecisionTreeClassifier) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = model.tree_
    n_features = model.n_features_in_
    rows: list[tuple[np.ndarray, np.ndarray, int]] = []

    def walk(node_id: int, low: np.ndarray, high: np.ndarray) -> None:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            rows.append((low.copy(), high.copy(), int(np.argmax(tree.value[node_id][0]))))
            return
        j = int(tree.feature[node_id])
        v = float(tree.threshold[node_id])
        left_high = high.copy()
        left_high[j] = min(left_high[j], v)
        walk(int(tree.children_left[node_id]), low, left_high)
        right_low = low.copy()
        right_low[j] = max(right_low[j], v)
        walk(int(tree.children_right[node_id]), right_low, high)

    walk(0, np.full(n_features, -np.inf), np.full(n_features, np.inf))
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def exact_continuous_tree_attack(
    model, X: np.ndarray, y: np.ndarray, binary_mask: np.ndarray, eps: float = 0.05
) -> dict:
    """Exact minimum-Linf attack over opposite-label leaves, holding binary columns fixed."""
    pred = np.asarray(model.predict(X), dtype=int)
    if isinstance(model, MarginAwareTree):
        lower, upper, leaf_class = enumerate_custom_leaves(model, X.shape[1])
    else:
        lower, upper, leaf_class = enumerate_sklearn_leaves(model)
    attacked = X.copy()
    min_distances = np.full(len(X), np.inf)
    for i, x in enumerate(X):
        need_low = np.maximum(lower - x[None, :] + 1e-9, 0.0)
        need_high = np.maximum(x[None, :] - upper, 0.0)
        required = np.maximum(need_low, need_high)
        invalid_binary = np.any(required[:, binary_mask] > 1e-8, axis=1) if np.any(binary_mask) else np.zeros(len(lower), dtype=bool)
        distances = np.max(required[:, ~binary_mask], axis=1) if np.any(~binary_mask) else np.zeros(len(lower))
        distances[(leaf_class == pred[i]) | invalid_binary] = np.inf
        leaf = int(np.argmin(distances))
        min_distances[i] = distances[leaf]
        if distances[leaf] <= eps:
            z = x.copy()
            cont = ~binary_mask
            z[cont] = np.maximum(z[cont], lower[leaf, cont] + 1e-9)
            z[cont] = np.minimum(z[cont], upper[leaf, cont])
            attacked[i] = z
    attacked_pred = np.asarray(model.predict(attacked), dtype=int)
    eligible = np.isfinite(min_distances)
    return {
        "eps": eps,
        "attack_success_rate": float(np.mean(attacked_pred != pred)),
        "attacked_accuracy": float(accuracy_score(y, attacked_pred)),
        "median_min_flip_linf": float(np.median(min_distances[eligible])) if np.any(eligible) else np.nan,
        "reachable_fraction": float(np.mean(eligible)),
    }


def evaluate_method(
    dataset: str,
    seed: int,
    method: str,
    model,
    fit_meta: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    binary_mask: np.ndarray,
    data_meta: dict,
    enable_attack: bool,
) -> tuple[list[dict], dict, dict | None]:
    rows = []
    for sigma in NOISE_LEVELS:
        X_eval = noisy_continuous_copy(X_test, sigma, seed + 1009 + int(1000 * sigma), binary_mask)
        pred = model.predict(X_eval)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "noise": sigma,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "fit_time_sec": float(fit_meta["fit_time_sec"]),
            "binary_feature_fraction": float(np.mean(binary_mask)),
            "categorical_raw_fraction": float(data_meta.get("n_categorical_features", 0)) / max(1, float(data_meta.get("n_raw_features", 1))),
        })
    stat_row = {"dataset": dataset, "seed": seed, **fit_meta}
    attack = None
    is_tree = isinstance(model, (MarginAwareTree, DecisionTreeClassifier))
    if enable_attack and is_tree:
        attack = {"dataset": dataset, "seed": seed, "method": method, **exact_continuous_tree_attack(model, X_test, y_test, binary_mask)}
    return rows, stat_row, attack


def run_one(
    dataset: str, seed: int, selected_methods: tuple[str, ...], enable_attacks: bool
) -> tuple[list[dict], list[dict], list[dict], dict | None, str | None]:
    set_thread_limits()
    try:
        spec = spec_for(dataset)
        X, y, data_meta = load_raw_dataset(spec, seed)
        stratify = y if np.min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.35, random_state=seed, stratify=stratify
        )
        X_train, X_test, binary_mask, data_meta = preprocess_train_test(X_train, X_test, data_meta)
        models: list[tuple[str, object, dict]] = []
        for method, alpha, policy in [
            ("matched_gini", 0.0, "standard"),
            ("mags_fixed", 0.5, "standard"),
            ("mags_binary_excluded", 0.5, "exclude"),
        ]:
            if method in selected_methods:
                model, meta = fit_custom(X_train, y_train, spec, method, alpha, binary_mask, policy)
                models.append((method, model, meta))
        if "cart_fixed" in selected_methods:
            started = time.perf_counter()
            cart = DecisionTreeClassifier(
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_samples_leaf,
                criterion="gini",
                random_state=seed,
            ).fit(X_train, y_train)
            meta = sklearn_stats(cart)
            meta.update(method="cart_fixed", fit_time_sec=time.perf_counter() - started)
            models.append(("cart_fixed", cart, meta))

        if "mags_tuned" in selected_methods:
            model, raw = fit_tuned_margin(X_train, y_train, spec, seed)
            meta = custom_stats(model)
            meta.update(method="mags_tuned", fit_time_sec=float(raw["fit_time_sec"]))
            models.append(("mags_tuned", model, meta))
        for method in ["cart_tuned_depth", "cart_pruned", "noise_augmented_cart", "random_forest", "extra_trees"]:
            if method not in selected_methods:
                continue
            model, raw = fit_sklearn_model(X_train, y_train, spec, method, seed)
            if isinstance(model, DecisionTreeClassifier):
                meta = sklearn_stats(model)
            else:
                meta = {"n_nodes": np.nan, "n_leaves": np.nan, "depth": np.nan, "mean_node_margin": np.nan, "binary_split_fraction": np.nan}
            meta.update(method=method, fit_time_sec=float(raw["fit_time_sec"]))
            models.append((method, model, meta))
        for method, kernel in [("linear_svm", "linear"), ("rbf_svm", "rbf")]:
            if method not in selected_methods:
                continue
            started = time.perf_counter()
            model = SVC(C=1.0, kernel=kernel, gamma="scale", random_state=seed).fit(X_train, y_train)
            meta = {
                "method": method,
                "fit_time_sec": time.perf_counter() - started,
                "n_nodes": np.nan,
                "n_leaves": np.nan,
                "depth": np.nan,
                "mean_node_margin": np.nan,
                "binary_split_fraction": np.nan,
            }
            models.append((method, model, meta))

        metric_rows: list[dict] = []
        stat_rows: list[dict] = []
        attack_rows: list[dict] = []
        for method, model, fit_meta in models:
            metrics, stats, attack = evaluate_method(
                dataset, seed, method, model, fit_meta, X_test, y_test, binary_mask, data_meta, enable_attacks
            )
            metric_rows.extend(metrics)
            stat_rows.append(stats)
            if attack is not None:
                attack_rows.append(attack)
        metadata = {
            "dataset": dataset,
            "seed": seed,
            **data_meta,
            "n_instances_used": int(len(y)),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "n_binary_encoded_features": int(np.sum(binary_mask)),
        }
        return metric_rows, stat_rows, attack_rows, metadata, None
    except Exception as exc:
        return [], [], [], None, f"{dataset} seed={seed}: {type(exc).__name__}: {exc}"


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().astype(float)
    m = float(values.mean())
    if len(values) < 2:
        return m, m, m
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return m, m - half, m + half


def paired_summary(df: pd.DataFrame, ours: str, base: str, noise: float) -> dict:
    sub = df[(df["noise"] == noise) & df["method"].isin([ours, base])]
    paired = sub.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    by_dataset = (paired[ours] - paired[base]).groupby(level="dataset").mean()
    # Preserve numerical ties after CSV round-trips; values at this scale are
    # floating-point subtraction artifacts, not measurable accuracy effects.
    by_dataset.loc[np.isclose(by_dataset, 0.0, atol=1e-12)] = 0.0
    m, lo, hi = mean_ci(by_dataset)
    p = float(wilcoxon(by_dataset, alternative="two-sided").pvalue) if np.any(by_dataset != 0) else 1.0
    return {
        "method": ours,
        "baseline": base,
        "noise": noise,
        "n_datasets": int(len(by_dataset)),
        "n_seeds": int(paired.index.get_level_values("seed").nunique()),
        "mean_delta": m,
        "ci95_low": lo,
        "ci95_high": hi,
        "median_delta": float(by_dataset.median()),
        "wins": int((by_dataset > 0).sum()),
        "ties": int((by_dataset == 0).sum()),
        "losses": int((by_dataset < 0).sum()),
        "wilcoxon_two_sided_p": p,
    }


def build_summaries(metrics: pd.DataFrame, stats: pd.DataFrame, attacks: pd.DataFrame) -> dict:
    comparisons = [
        paired_summary(metrics, "mags_fixed", "matched_gini", 0.20),
        paired_summary(metrics, "mags_binary_excluded", "matched_gini", 0.20),
        paired_summary(metrics, "mags_binary_excluded", "mags_fixed", 0.20),
        paired_summary(metrics, "mags_tuned", "cart_tuned_depth", 0.20),
        paired_summary(metrics, "mags_tuned", "cart_pruned", 0.20),
        paired_summary(metrics, "mags_tuned", "noise_augmented_cart", 0.20),
    ]
    accuracy = []
    focus = metrics[metrics["noise"] == 0.20]
    for method, group in focus.groupby("method"):
        dataset_means = group.groupby("dataset")["accuracy"].mean()
        m, lo, hi = mean_ci(dataset_means)
        accuracy.append({"method": method, "mean_accuracy": m, "ci95_low": lo, "ci95_high": hi, "n_datasets": int(len(dataset_means)), "n_seeds": int(group.seed.nunique())})
    attack_summary = []
    if not attacks.empty:
        for method, group in attacks.groupby("method"):
            dataset_means = group.groupby("dataset")[["attack_success_rate", "attacked_accuracy", "median_min_flip_linf"]].mean()
            attack_summary.append({
                "method": method,
                "n_datasets": int(len(dataset_means)),
                "n_seeds": int(group.seed.nunique()),
                "attack_success_rate": float(dataset_means["attack_success_rate"].mean()),
                "attacked_accuracy": float(dataset_means["attacked_accuracy"].mean()),
                "median_min_flip_linf": float(dataset_means["median_min_flip_linf"].median()),
            })
    fixed = focus[focus.method.isin(["mags_fixed", "matched_gini"])]
    pivot = fixed.pivot(index=["dataset", "seed"], columns="method", values="accuracy").dropna()
    deltas = (pivot.mags_fixed - pivot.matched_gini).groupby(level="dataset").mean()
    deltas.loc[np.isclose(deltas, 0.0, atol=1e-12)] = 0.0
    covariates = focus[focus.method == "mags_fixed"].groupby("dataset")[["categorical_raw_fraction", "binary_feature_fraction"]].mean()
    joined = covariates.join(deltas.rename("delta")).dropna()
    cat_rho, cat_p = spearmanr(joined.categorical_raw_fraction, joined.delta)
    bin_rho, bin_p = spearmanr(joined.binary_feature_fraction, joined.delta)
    split_summary = []
    for method in ["matched_gini", "mags_fixed", "mags_binary_excluded"]:
        g = stats[stats.method == method].groupby("dataset")["binary_split_fraction"].mean()
        split_summary.append({"method": method, "mean_binary_split_fraction": float(g.mean()), "n_datasets": int(len(g))})
    return {
        "paired_comparisons": comparisons,
        "accuracy_at_sigma_02": accuracy,
        "exact_attack_eps_005": attack_summary,
        "representation_effect": {
            "categorical_fraction_spearman_rho": float(cat_rho),
            "categorical_fraction_p": float(cat_p),
            "binary_fraction_spearman_rho": float(bin_rho),
            "binary_fraction_p": float(bin_p),
            "binary_split_summary": split_summary,
        },
    }


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(COMPLETE_DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, range(30))))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method-group", choices=["all", "core", "baselines", "trees"], default="all")
    parser.add_argument("--attacks", choices=["all", "none"], default="all")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    selected_methods = {
        "all": tuple(CORE_METHODS + EXTENDED_METHODS),
        "core": tuple(CORE_METHODS),
        "baselines": tuple(EXTENDED_METHODS),
        "trees": tuple(TREE_METHODS),
    }[args.method_group]
    enable_attacks = args.attacks == "all"
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    metrics: list[dict] = []
    stats: list[dict] = []
    attacks: list[dict] = []
    metadata: list[dict] = []
    failures: list[str] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, dataset, seed, selected_methods, enable_attacks): (dataset, seed)
            for dataset, seed in jobs
        }
        for future in as_completed(futures):
            dataset, seed = futures[future]
            m, s, a, meta, failure = future.result()
            if failure:
                failures.append(failure)
                print(f"[failed] {failure}", flush=True)
            else:
                print(f"[done] {dataset} seed={seed}", flush=True)
            metrics.extend(m)
            stats.extend(s)
            attacks.extend(a)
            if meta is not None:
                metadata.append(meta)
    metrics_df = pd.DataFrame(metrics)
    stats_df = pd.DataFrame(stats)
    attacks_df = pd.DataFrame(attacks)
    metrics_df.to_csv(OUT / "results.csv", index=False)
    stats_df.to_csv(OUT / "tree_stats.csv", index=False)
    attacks_df.to_csv(OUT / "exact_attacks.csv", index=False)
    if set(CORE_METHODS + EXTENDED_METHODS).issubset(set(metrics_df.get("method", []))):
        summary = build_summaries(metrics_df, stats_df, attacks_df)
    else:
        summary = {
            "stage": args.method_group,
            "selected_methods": list(selected_methods),
            "metrics_rows": int(len(metrics_df)),
            "tree_stats_rows": int(len(stats_df)),
            "attack_rows": int(len(attacks_df)),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": time.perf_counter() - started,
        "datasets": datasets,
        "seeds": seeds,
        "failures": failures,
        "completed_jobs": len(metadata),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "method_protocol": {
            "selected_methods": list(selected_methods),
            "all_methods_seeds": len(seeds),
            "exact_attack_tree_methods_seeds": len(seeds) if enable_attacks else 0,
            "noise": "Gaussian noise on train-standardized non-binary encoded columns only",
            "attack": "Exact minimum-Linf opposite-leaf attack with binary columns fixed",
        },
        "dataset_metadata": metadata,
    }
    (OUT / "metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(metadata), "failures": len(failures), "elapsed_sec": run_meta["elapsed_sec"]}, indent=2))
    return 0 if len(metadata) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
