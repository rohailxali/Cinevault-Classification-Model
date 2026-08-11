#!/usr/bin/env python
"""
Cinevault Classification Model -- Training Entry Point
=======================================================

Trains and evaluates four models (DummyClassifier, LogisticRegression,
DecisionTree, RandomForest) under two configurations:

  Config B (main): duration excluded -- genuine model comparison.
  Config A (demo): duration_is_minutes included -- shows near-trivial
                   ceiling caused by the unit-encoding of the target.

Model selection criterion (stated before any results are examined):
  Primary  : macro F1 on 5-fold stratified CV (training set only).
  Tiebreak : ROC-AUC.
Rationale: 70/30 class imbalance makes raw accuracy misleading; macro F1
weights both classes equally and surfaces minority-class performance.

The winning Config B model is saved to models/cinevault_classifier_v1.joblib.
Reports are saved to reports/.

Usage
-----
    python train.py
"""

import sys
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    CLASS_NAMES,
    CV_FOLDS,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_FILE,
    REPORTS_DIR,
    SELECTION_METRIC,
    TARGET_COL,
    TARGET_MAP,
    TEST_SIZE,
)
from src.data.loader import load_raw
from src.evaluation.metrics import build_results_table, compute_test_metrics
from src.models.trainer import PARAM_GRID_RF, build_models
from src.utils.io import save_pipeline

# Source dataset (sits at project root alongside train.py)
DATASET_SRC = PROJECT_ROOT / "Dataset.csv"

_DIVIDER = "=" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_directories() -> None:
    for d in [
        RAW_DATA_FILE.parent,
        PROJECT_ROOT / "data" / "processed",
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def _copy_dataset() -> None:
    if not RAW_DATA_FILE.exists():
        if not DATASET_SRC.exists():
            raise FileNotFoundError(
                f"Source dataset not found at '{DATASET_SRC}'.\n"
                "Please ensure Dataset.csv is in the project root."
            )
        shutil.copy2(DATASET_SRC, RAW_DATA_FILE)
        print(f"  Copied dataset -> {RAW_DATA_FILE}")
    else:
        print(f"  Dataset already present at {RAW_DATA_FILE}")


def _run_config(
    label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    include_duration: bool,
    run_grid_search: bool,
) -> tuple:
    """
    Full train-CV-evaluate cycle for one configuration.

    Returns
    -------
    (fitted_models dict, cv_scores dict, results_df, winner_name, tuned_pipeline_or_None)
    """
    print(f"\n{_DIVIDER}")
    print(f"  {label}")
    print(_DIVIDER)

    models = build_models(include_duration=include_duration)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Cross-validation (training set only) ──────────────────────────────
    print(f"\n  Cross-validation ({CV_FOLDS}-fold stratified | metric: {SELECTION_METRIC}):")
    cv_scores: dict = {}
    for name, pipe in models.items():
        scores = cross_val_score(
            clone(pipe), X_train, y_train, cv=skf, scoring=SELECTION_METRIC
        )
        cv_scores[name] = scores
        print(f"    {name:<28}  {SELECTION_METRIC} = {scores.mean():.4f} +/- {scores.std():.4f}")

    # ── Winner selection (stated criterion, non-Dummy only) ───────────────
    real_scores = {k: v for k, v in cv_scores.items() if k != "DummyClassifier"}
    winner_name = max(real_scores, key=lambda k: real_scores[k].mean())
    print(f"\n  Winner by CV {SELECTION_METRIC}: {winner_name}")

    # ── Optional grid search on the winner (Config B only) ────────────────
    tuned_pipeline = None
    if run_grid_search and winner_name == "RandomForest":
        print(f"\n  Grid search on {winner_name}  (grid size = {len(PARAM_GRID_RF['clf__n_estimators'])} x {len(PARAM_GRID_RF['clf__max_depth'])} = {len(PARAM_GRID_RF['clf__n_estimators'])*len(PARAM_GRID_RF['clf__max_depth'])} combos x {CV_FOLDS} folds) ...")
        gs = GridSearchCV(
            clone(models[winner_name]),
            PARAM_GRID_RF,
            cv=skf,
            scoring=SELECTION_METRIC,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        tuned_pipeline = gs.best_estimator_
        print(f"  Best params  : {gs.best_params_}")
        print(f"  Best CV F1   : {gs.best_score_:.4f}")

    # ── Fit all base models on full training set ──────────────────────────
    print("\n  Fitting final models on full training set ...")
    fitted: dict = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    if tuned_pipeline is not None:
        fitted["RandomForest_tuned"] = tuned_pipeline

    # ── Evaluate on test set (touched exactly once) ────────────────────────
    print("\n  Test-set evaluation:")
    all_metrics: dict = {}
    for name, pipe in fitted.items():
        m = compute_test_metrics(pipe, X_test, y_test)
        cv_f1 = cv_scores.get(name.replace("_tuned", ""), np.array([float("nan")])).mean()
        m["cv_f1_macro"] = cv_f1
        all_metrics[name] = m

    results_df = build_results_table(all_metrics)
    print("\n" + results_df.to_string(index=False))

    # ── Per-model classification reports ──────────────────────────────────
    print(f"\n  Classification reports (test set):")
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        print(f"\n  --- {name} ---")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return fitted, cv_scores, results_df, winner_name, tuned_pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(_DIVIDER)
    print("  Cinevault Classification Model -- Training Pipeline")
    print(f"  Random seed  : {RANDOM_STATE}")
    print(f"  Test size    : {TEST_SIZE:.0%}")
    print(f"  CV folds     : {CV_FOLDS}")
    print(f"  Select by    : {SELECTION_METRIC} (tiebreak: roc_auc)")
    print(_DIVIDER)

    _setup_directories()
    _copy_dataset()

    # ── Load & report ──────────────────────────────────────────────────────
    df = load_raw(RAW_DATA_FILE)
    total = len(df)
    print(f"\n  Dataset: {total} rows x {df.shape[1]} columns")
    print("  Target distribution:")
    for cls, cnt in df[TARGET_COL].value_counts().items():
        print(f"    {cls:<10} {cnt:>5} ({cnt / total * 100:.1f}%)")

    # ── Encode target ──────────────────────────────────────────────────────
    y = df[TARGET_COL].map(TARGET_MAP)
    X = df.drop(columns=[TARGET_COL])

    # ── Stratified 80/20 split ─────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Config B -- main comparison (no duration) ──────────────────────────
    fitted_b, cv_b, results_b, winner_b, tuned_b = _run_config(
        label="CONFIG B -- No duration feature  (main / authoritative comparison)",
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        include_duration=False,
        run_grid_search=True,
    )

    # ── Config A -- leakage ceiling (with duration_is_minutes) ────────────
    fitted_a, cv_a, results_a, winner_a, _ = _run_config(
        label="CONFIG A -- With duration_is_minutes  (leakage ceiling demonstration)",
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        include_duration=True,
        run_grid_search=False,
    )

    # ── Save reports ───────────────────────────────────────────────────────
    results_b.to_csv(REPORTS_DIR / "results_config_b.csv", index=False)
    results_a.to_csv(REPORTS_DIR / "results_config_a.csv", index=False)
    print(f"\n  Reports saved to {REPORTS_DIR}/")

    # ── Save best Config B pipeline ────────────────────────────────────────
    if tuned_b is not None:
        best_name = "RandomForest_tuned"
    else:
        best_name = winner_b
    best_pipeline = fitted_b[best_name]
    model_path = save_pipeline(best_pipeline, "cinevault_classifier", version="v1")
    print(f"  Winning model ({best_name}) saved to {model_path}")

    print(f"\n{_DIVIDER}")
    print("  Training complete.")
    print(_DIVIDER)


if __name__ == "__main__":
    main()
