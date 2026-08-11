"""
Test suite for the Netflix content-type classifier pipeline.

Coverage:
  - Dataset loading and schema validation
  - Target column values and encoding
  - Feature engineering (output shape, column names, binary values, NaN-free)
  - Duration config A / config B column presence
  - Full pipeline fit/predict for all four model types
  - Unseen category graceful handling (rating, country)
  - Model artifact save -> reload -> same predictions
  - Pipeline handles rows with Not Given fields

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TARGET_MAP,
)
from src.data.loader import REQUIRED_COLUMNS, TARGET_VALUES, load_raw
from src.features.engineer import NetflixFeatureEngineer
from src.models.trainer import build_models
from src.utils.io import load_pipeline


# ---------------------------------------------------------------------------
# Module-scoped fixtures (load once for the whole test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_df():
    return load_raw(RAW_DATA_FILE)


@pytest.fixture(scope="module")
def small_sample(raw_df):
    """Balanced 100-row sample for fast test runs."""
    movie = raw_df[raw_df[TARGET_COL] == "Movie"].head(50)
    tv    = raw_df[raw_df[TARGET_COL] == "TV Show"].head(50)
    return pd.concat([movie, tv]).reset_index(drop=True)


@pytest.fixture(scope="module")
def Xy(small_sample):
    y = small_sample[TARGET_COL].map(TARGET_MAP)
    X = small_sample.drop(columns=[TARGET_COL])
    return X, y


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    def test_loads_without_error(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_row_count(self, raw_df):
        assert 8000 < len(raw_df) < 10_000, f"Unexpected row count: {len(raw_df)}"

    def test_required_columns_present(self, raw_df):
        for col in REQUIRED_COLUMNS:
            assert col in raw_df.columns, f"Column '{col}' missing from dataset"

    def test_no_true_nulls(self, raw_df):
        null_sum = raw_df.isnull().sum().sum()
        assert null_sum == 0, f"Expected 0 NaN values, got {null_sum}"

    def test_no_duplicate_rows(self, raw_df):
        dupes = raw_df.duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate rows"


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------


class TestTargetColumn:
    def test_exact_target_values(self, raw_df):
        actual = set(raw_df[TARGET_COL].unique())
        assert actual == TARGET_VALUES, f"Unexpected values: {actual}"

    def test_movie_encodes_to_1(self, raw_df):
        movie_rows = raw_df[raw_df[TARGET_COL] == "Movie"]
        assert movie_rows[TARGET_COL].map(TARGET_MAP).iloc[0] == 1

    def test_tvshow_encodes_to_0(self, raw_df):
        tv_rows = raw_df[raw_df[TARGET_COL] == "TV Show"]
        assert tv_rows[TARGET_COL].map(TARGET_MAP).iloc[0] == 0

    def test_class_balance_approx_70_30(self, raw_df):
        movie_pct = (raw_df[TARGET_COL] == "Movie").mean()
        assert 0.65 < movie_pct < 0.75, f"Movie pct: {movie_pct:.2%}"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_fit_transform_runs(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer()
        out = eng.fit(X).transform(X)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(X)

    def test_expected_base_columns_present(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer(include_duration=False)
        out = eng.fit_transform(X)
        expected = {
            "has_director", "country_primary", "year_added", "month_added",
            "years_since_release", "release_year", "rating_grouped",
            "primary_genre", "genre_count",
        }
        missing = expected - set(out.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_duration_absent_config_b(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer(include_duration=False)
        out = eng.fit_transform(X)
        assert "duration_is_minutes" not in out.columns

    def test_duration_present_config_a(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer(include_duration=True)
        out = eng.fit_transform(X)
        assert "duration_is_minutes" in out.columns
        assert set(out["duration_is_minutes"].unique()).issubset({0, 1})

    def test_has_director_is_binary(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer()
        out = eng.fit_transform(X)
        assert set(out["has_director"].unique()).issubset({0, 1})

    def test_non_nullable_columns_have_no_nan(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer()
        out = eng.fit_transform(X)
        non_nullable = ["has_director", "release_year", "genre_count",
                        "country_primary", "primary_genre", "rating_grouped"]
        for col in non_nullable:
            assert out[col].isnull().sum() == 0, f"NaN found in '{col}'"

    def test_top_genres_learned_on_fit(self, Xy):
        X, _ = Xy
        # Use min_genre_count=1 so the attribute is non-empty even on the
        # 100-row test fixture (no genre hits >=20 at that sample size).
        eng = NetflixFeatureEngineer(min_genre_count=1)
        assert not hasattr(eng, "top_genres_")
        eng.fit(X)
        assert hasattr(eng, "top_genres_")
        assert isinstance(eng.top_genres_, set)
        assert len(eng.top_genres_) > 0

    def test_top_countries_learned_on_fit(self, Xy):
        X, _ = Xy
        eng = NetflixFeatureEngineer()
        eng.fit(X)
        assert hasattr(eng, "top_countries_")
        assert isinstance(eng.top_countries_, set)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


# Module-scoped fixture (outside the class) avoids the pytest deprecation
# warning about class-scoped fixtures defined as instance methods.
@pytest.fixture(scope="module")
def fitted_lr(Xy):
    X, y = Xy
    pipe = build_models(include_duration=False)["LogisticRegression"]
    pipe.fit(X, y)
    return pipe, X, y


class TestPipeline:
    def test_predict_returns_correct_length(self, fitted_lr):
        pipe, X, y = fitted_lr
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_predict_only_valid_labels(self, fitted_lr):
        pipe, X, _ = fitted_lr
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, fitted_lr):
        pipe, X, _ = fitted_lr
        probs = pipe.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_all_four_models_train_and_predict(self, Xy):
        X, y = Xy
        for name, pipe in build_models(include_duration=False).items():
            pipe.fit(X, y)
            preds = pipe.predict(X)
            assert len(preds) == len(y), f"{name}: predict length mismatch"
            assert set(preds).issubset({0, 1}), f"{name}: unexpected label values"

    def test_unseen_rating_no_crash(self, Xy):
        X, y = Xy
        pipe = build_models()["LogisticRegression"]
        pipe.fit(X, y)
        row = X.iloc[[0]].copy()
        row["rating"] = "COMPLETELY-UNSEEN-RATING"
        pred = pipe.predict(row)
        assert len(pred) == 1

    def test_unseen_country_no_crash(self, Xy):
        X, y = Xy
        pipe = build_models()["RandomForest"]
        pipe.fit(X, y)
        row = X.iloc[[0]].copy()
        row["country"] = "PlanetZorgnon99"
        pred = pipe.predict(row)
        assert len(pred) == 1

    def test_not_given_director_handled(self, Xy):
        X, y = Xy
        pipe = build_models()["LogisticRegression"]
        pipe.fit(X, y)
        row = X.iloc[[0]].copy()
        row["director"] = "Not Given"
        pred = pipe.predict(row)
        assert len(pred) == 1


# ---------------------------------------------------------------------------
# Model artifact persistence
# ---------------------------------------------------------------------------


class TestModelArtifact:
    def test_save_and_reload_same_predictions(self, Xy, tmp_path):
        X, y = Xy
        pipe = build_models()["LogisticRegression"]
        pipe.fit(X, y)

        save_path = tmp_path / "test_lr_v1.joblib"
        joblib.dump(pipe, save_path)
        loaded = load_pipeline(save_path)

        preds_orig   = pipe.predict(X)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded), "Loaded model predictions differ"

    def test_loaded_model_handles_unseen_category(self, Xy, tmp_path):
        X, y = Xy
        pipe = build_models()["LogisticRegression"]
        pipe.fit(X, y)

        save_path = tmp_path / "test_lr_unseen.joblib"
        joblib.dump(pipe, save_path)
        loaded = load_pipeline(save_path)

        row = X.iloc[[0]].copy()
        row["rating"] = "UNSEEN-XYZ-9999"
        pred = loaded.predict(row)
        assert len(pred) == 1

    def test_loaded_model_predict_proba(self, Xy, tmp_path):
        X, y = Xy
        pipe = build_models()["RandomForest"]
        pipe.fit(X, y)

        save_path = tmp_path / "test_rf.joblib"
        joblib.dump(pipe, save_path)
        loaded = load_pipeline(save_path)

        probs = loaded.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
