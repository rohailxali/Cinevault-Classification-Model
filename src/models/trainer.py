"""
Pipeline builder for the Netflix content-type classifier.

Each model is wrapped in a three-step sklearn Pipeline:
    NetflixFeatureEngineer -> ColumnTransformer -> Estimator

The full pipeline takes a raw DataFrame (same shape as the CSV minus the
target column) and returns predictions directly — no manual preprocessing
steps live outside the pipeline.

Preprocessing note — shared scaler tradeoff
-------------------------------------------
Tree-based models (DecisionTree, RandomForest) do not require feature
scaling, but a single shared ColumnTransformer (including StandardScaler
for numeric features) is used for simplicity.  The scaler is a monotone
transform and does not change split decisions in trees, so it causes no
harm — it only adds a tiny amount of computation.  This tradeoff is
acceptable here and is explicitly documented.

class_weight="balanced" handles the 70/30 Movie/TV Show imbalance without
introducing an oversampling library, which would be unnecessary complexity
at this data scale.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE
from src.features.engineer import NetflixFeatureEngineer

# ── Column groups (must match NetflixFeatureEngineer output columns) ──────────
NUMERIC_FEATURES = [
    "release_year",
    "year_added",
    "month_added",
    "years_since_release",
    "genre_count",
]
BINARY_FEATURES_BASE = ["has_director"]
CATEGORICAL_FEATURES = ["country_primary", "primary_genre", "rating_grouped"]
DURATION_FEATURE     = "duration_is_minutes"   # added only in Config A

# ── Hyperparameter grid (Random Forest only, conservative size) ───────────────
# 2 values x 3 values = 6 param combinations x 5 CV folds = 30 total fits.
PARAM_GRID_RF = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth":    [None, 10, 20],
}


def build_preprocessor(include_duration: bool = False) -> ColumnTransformer:
    """
    Build the ColumnTransformer for one configuration.

    Parameters
    ----------
    include_duration : bool
        If True, passthrough ``duration_is_minutes`` as a binary feature.
    """
    binary_features = BINARY_FEATURES_BASE + (
        [DURATION_FEATURE] if include_duration else []
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" -> unseen categories at inference time
        # produce an all-zero row rather than raising an error.
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,    NUMERIC_FEATURES),
            ("bin", "passthrough",       binary_features),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _make_pipeline(estimator, include_duration: bool) -> Pipeline:
    """Wrap feature engineer + preprocessor + estimator into one Pipeline."""
    return Pipeline([
        ("engineer",     NetflixFeatureEngineer(include_duration=include_duration)),
        ("preprocessor", build_preprocessor(include_duration=include_duration)),
        ("clf",          estimator),
    ])


def build_models(include_duration: bool = False) -> dict:
    """
    Return a fresh dict of name -> unfitted Pipeline for all four models.

    Each call creates new instances, so pipelines can be cloned or fitted
    independently without sharing state.

    Parameters
    ----------
    include_duration : bool
        Passed to NetflixFeatureEngineer and build_preprocessor.
    """
    return {
        "DummyClassifier": _make_pipeline(
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
            include_duration,
        ),
        "LogisticRegression": _make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            include_duration,
        ),
        "DecisionTree": _make_pipeline(
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            include_duration,
        ),
        "RandomForest": _make_pipeline(
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            include_duration,
        ),
    }
