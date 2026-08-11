"""
Central configuration for the Netflix content-type classifier.

All paths are derived from this file's location so the project is
portable across machines (no hard-coded absolute paths).

Random seed is fixed here and imported everywhere else — changing it
in one place propagates consistently across split, CV, and model init.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Directory layout ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "models"
REPORTS_DIR        = PROJECT_ROOT / "reports"

RAW_DATA_FILE = DATA_RAW_DIR / "Dataset.csv"

# ── Target column & encoding ─────────────────────────────────────────────────
TARGET_COL   = "type"
TARGET_MAP   = {"Movie": 1, "TV Show": 0}   # Movie = positive class
CLASS_NAMES  = ["TV Show", "Movie"]          # index 0 -> TV Show, 1 -> Movie

# ── Feature engineering thresholds ──────────────────────────────────────────
# Countries with fewer occurrences than this in training data -> "Other"
RARE_COUNTRY_THRESHOLD: int = 50
# Primary genres with fewer occurrences than this in training data -> "Other"
MIN_GENRE_COUNT: int = 20

# ── CV / evaluation ──────────────────────────────────────────────────────────
CV_FOLDS: int = 5
TEST_SIZE: float = 0.20

# ── Model selection criterion ─────────────────────────────────────────────────
# Primary: macro F1 on CV folds (training set only).
# Rationale: 70/30 class imbalance makes accuracy unreliable; macro F1
# weights both classes equally, surfacing minority-class (TV Show) performance.
# Tiebreaker: ROC-AUC (threshold-independent).
SELECTION_METRIC = "f1_macro"
