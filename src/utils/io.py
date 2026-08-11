"""
Model persistence utilities.

Wraps joblib save/load with versioned filenames so multiple model
artifacts can coexist without overwriting each other.
"""

import joblib
from pathlib import Path

from src.config import MODELS_DIR


def save_pipeline(pipeline, name: str, version: str = "v1") -> Path:
    """
    Serialize a fitted pipeline to disk.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    name     : descriptive artifact name (e.g. "cinevault_classifier")
    version  : version tag appended to the filename

    Returns
    -------
    Path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}_{version}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path) -> object:
    """
    Load a pipeline artifact from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    The deserialized sklearn Pipeline object.
    """
    return joblib.load(Path(path))
