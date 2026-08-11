"""
Feature engineering for the Netflix content-type classifier.

Implemented as a scikit-learn-compatible transformer so it can sit
inside a Pipeline and be fit strictly on training data in every context
(initial split and inside each CV fold during hyperparameter search).

Feature decisions
-----------------
DROPPED (handled upstream — not passed to this transformer):
  show_id   — unique row identifier, zero generalizable signal
  title     — near-unique free text (8 787 unique values), no signal

BINARY (engineer before use):
  director  -> has_director (1 if populated, 0 if "Not Given")
               ~29.4% of rows have "Not Given"; raw name is too
               high-cardinality to one-hot encode.

HIGH-CARDINALITY CATEGORICALS (engineer before OHE):
  country   -> country_primary: first-listed country, rare values
               (< RARE_COUNTRY_THRESHOLD) bucketed to "Other".
               "Not Given" (287 rows) kept as its own category because
               it has predictive content (prevalence differs by type).
  listed_in -> primary_genre: first genre in the comma list; genre_count:
               number of genres listed.  513 unique combinations reduced
               to top genres + "Other".

DATE-DERIVED NUMERICS:
  date_added -> year_added, month_added, years_since_release
                (year_added - release_year).  All numeric, imputed by
                median downstream if parsing yields NaN.

DIRECT NUMERIC (kept as-is):
  release_year — passed through unchanged.

LOW-CARDINALITY CATEGORICAL (standard OHE):
  rating     — 14 values; 3 rare ones (TV-Y7-FV, NC-17, UR) grouped
               into "Rare" to avoid sparse dummies.

DURATION — dual-configuration (see README for full rationale):
  Config A (include_duration=True):
    duration_is_minutes = 1 if "min" in the string, else 0.
    In this dataset every Movie has "X min" and every TV Show has
    "X Season(s)" — zero crossover.  This is a near-perfect label
    leak.  Config A exists only to demonstrate the ceiling effect.
  Config B (include_duration=False):
    Duration excluded entirely.  This is the authoritative comparison.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NOT_GIVEN = "Not Given"


class NetflixFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that engineers all features from raw columns.

    Parameters
    ----------
    include_duration : bool
        If True, add ``duration_is_minutes`` (Config A / leakage demo).
        If False (default), exclude it (Config B / real comparison).
    rare_country_threshold : int
        Countries with fewer occurrences than this in the training data
        are bucketed to "Other".
    min_genre_count : int
        Primary genres with fewer occurrences than this in the training
        data are bucketed to "Other".
    """

    def __init__(
        self,
        include_duration: bool = False,
        rare_country_threshold: int = 50,
        min_genre_count: int = 20,
    ):
        self.include_duration = include_duration
        self.rare_country_threshold = rare_country_threshold
        self.min_genre_count = min_genre_count

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "NetflixFeatureEngineer":
        """
        Learn top genres and top countries from training data.

        Both thresholds are fitted here so that transform() applies the
        same buckets to validation/test data without peeking at their
        distribution.
        """
        primary_genres = X["listed_in"].apply(self._extract_primary_genre)
        gc = primary_genres.value_counts()
        self.top_genres_: set = set(gc[gc >= self.min_genre_count].index)

        country_primary = X["country"].apply(self._extract_primary_country)
        cc = country_primary.value_counts()
        self.top_countries_: set = set(cc[cc >= self.rare_country_threshold].index)

        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return engineered feature DataFrame with named columns."""
        out = pd.DataFrame(index=X.index)

        # 1. has_director — binary missingness indicator
        out["has_director"] = (X["director"] != NOT_GIVEN).astype(int)

        # 2. country_primary — bucketed to top countries
        cp = X["country"].apply(self._extract_primary_country)
        out["country_primary"] = cp.apply(
            lambda x: x if x in self.top_countries_ else "Other"
        )

        # 3. date_added derived features
        dates = pd.to_datetime(X["date_added"], format="%m/%d/%Y", errors="coerce")
        out["year_added"]  = dates.dt.year
        out["month_added"] = dates.dt.month

        # 4. years_since_release (engineered temporal feature)
        out["years_since_release"] = out["year_added"] - X["release_year"]

        # 5. release_year — direct numeric
        out["release_year"] = X["release_year"]

        # 6. rating_grouped — merge thin tail into "Rare"
        rare_ratings = {"TV-Y7-FV", "NC-17", "UR"}
        out["rating_grouped"] = X["rating"].apply(
            lambda r: "Rare" if r in rare_ratings else r
        )

        # 7. primary_genre — bucketed to top genres
        pg = X["listed_in"].apply(self._extract_primary_genre)
        out["primary_genre"] = pg.apply(
            lambda g: g if g in self.top_genres_ else "Other"
        )

        # 8. genre_count — how many genres listed
        out["genre_count"] = X["listed_in"].apply(
            lambda s: len(s.split(",")) if isinstance(s, str) and s != NOT_GIVEN else 0
        )

        # 9. duration_is_minutes — Config A only
        if self.include_duration:
            out["duration_is_minutes"] = X["duration"].apply(
                lambda d: 1 if isinstance(d, str) and "min" in d else 0
            )

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_primary_genre(listed_in: str) -> str:
        """Return first genre from a comma-separated list."""
        if not isinstance(listed_in, str) or listed_in == NOT_GIVEN:
            return NOT_GIVEN
        return listed_in.split(",")[0].strip()

    @staticmethod
    def _extract_primary_country(country: str) -> str:
        """Return first country from a comma-separated list."""
        if not isinstance(country, str) or country == NOT_GIVEN:
            return NOT_GIVEN
        return country.split(",")[0].strip()
