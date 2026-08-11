"""
Data loading and schema validation for the Netflix titles dataset.

Treats the raw CSV as read-only — this module only reads, never writes
to the source file.
"""

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "show_id", "type", "title", "director", "country",
    "date_added", "release_year", "rating", "duration", "listed_in",
]
TARGET_VALUES = {"Movie", "TV Show"}


def load_raw(path: Path) -> pd.DataFrame:
    """
    Load and minimally validate the raw Netflix CSV.

    Parameters
    ----------
    path : Path
        Absolute path to Dataset.csv

    Returns
    -------
    pd.DataFrame — raw dataframe, untransformed.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if required columns or target values are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place Dataset.csv in data/raw/ or run train.py which copies it automatically."
        )

    df = pd.read_csv(path)

    # Schema check
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Target integrity check
    actual = set(df["type"].unique())
    unexpected = actual - TARGET_VALUES
    if unexpected:
        raise ValueError(
            f"Unexpected values in 'type' column: {unexpected}. "
            f"Expected only {TARGET_VALUES}."
        )

    return df
