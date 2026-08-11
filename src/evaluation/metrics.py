"""
Evaluation metrics for the Netflix content-type classifier.

Computes a full metrics dictionary for any fitted Pipeline evaluated on
a held-out test set.  All computations are self-contained — no global
state is modified.

Why these metrics?
------------------
accuracy     : Useful as a sanity check, but misleading in isolation under
               class imbalance.  A trivial classifier that always predicts
               "Movie" achieves ~70% accuracy on this dataset.
precision    : Of all titles predicted as X, how many actually are X?
               Matters when false positives are costly.
recall       : Of all actual X titles, how many did we catch?
               Matters when false negatives are costly.
F1 macro     : Harmonic mean of precision and recall, averaged equally
               across both classes — our primary selection metric because
               it gives equal weight to the minority class (TV Show).
ROC-AUC      : Area under the receiver-operating-characteristic curve;
               threshold-independent.  Used as tiebreaker.
confusion matrix : Raw counts; reveals directional error patterns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_test_metrics(pipeline, X_test: pd.DataFrame, y_test) -> dict:
    """
    Compute the full evaluation metrics for a fitted pipeline on the test set.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test   : raw feature DataFrame (same shape as training X)
    y_test   : true integer labels (Movie=1, TV Show=0)

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_movie, recall_movie, f1_movie,
        precision_tvshow, recall_tvshow, f1_tvshow,
        roc_auc, confusion_matrix, classification_report_str
    """
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")

    report_str = classification_report(
        y_test, y_pred, target_names=["TV Show", "Movie"]
    )

    return {
        "accuracy":          accuracy_score(y_test, y_pred),
        "precision_macro":   precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":      recall_score(y_test, y_pred,    average="macro", zero_division=0),
        "f1_macro":          f1_score(y_test, y_pred,        average="macro", zero_division=0),
        # Per-class (Movie = pos_label=1, TV Show = pos_label=0)
        "precision_movie":   precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_movie":      recall_score(y_test, y_pred,    pos_label=1, zero_division=0),
        "f1_movie":          f1_score(y_test, y_pred,        pos_label=1, zero_division=0),
        "precision_tvshow":  precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall_tvshow":     recall_score(y_test, y_pred,    pos_label=0, zero_division=0),
        "f1_tvshow":         f1_score(y_test, y_pred,        pos_label=0, zero_division=0),
        "roc_auc":           roc_auc,
        "confusion_matrix":  confusion_matrix(y_test, y_pred),
        "classification_report_str": report_str,
    }


def build_results_table(all_results: dict) -> pd.DataFrame:
    """
    Convert a dict of {model_name: metrics_dict} into a clean comparison DataFrame.

    Columns: Model, CV_F1_macro, Test_Accuracy, Test_F1_macro, Test_ROC_AUC,
             Test_Precision_macro, Test_Recall_macro, F1_Movie, F1_TVShow
    """
    rows = []
    for name, m in all_results.items():
        rows.append({
            "Model":                name,
            "CV_F1_macro":          round(m.get("cv_f1_macro",      float("nan")), 4),
            "Test_Accuracy":        round(m["accuracy"],             4),
            "Test_F1_macro":        round(m["f1_macro"],             4),
            "Test_ROC_AUC":         round(m["roc_auc"],              4),
            "Test_Precision_macro": round(m["precision_macro"],      4),
            "Test_Recall_macro":    round(m["recall_macro"],         4),
            "F1_Movie":             round(m["f1_movie"],             4),
            "F1_TVShow":            round(m["f1_tvshow"],            4),
        })
    return pd.DataFrame(rows)
