#!/usr/bin/env python
"""
Cinevault Classification Model -- Inference CLI
================================================

Loads the saved model artifact and predicts whether a Netflix title
is a Movie or TV Show given its raw metadata fields.

The pipeline bundles the full preprocessing (feature engineering +
column transformer) and the trained estimator, so this script does
not require any manual preprocessing steps.

Usage
-----
Minimal (required args only):
    python predict.py --date_added "7/16/2010" --release_year 2010 \
        --rating "PG-13" --duration "148 min" --listed_in "Action & Adventure"

Full example (Movie):
    python predict.py \
        --director "Christopher Nolan" \
        --country "United States" \
        --date_added "7/16/2010" \
        --release_year 2010 \
        --rating "PG-13" \
        --duration "148 min" \
        --listed_in "Action & Adventure, Sci-Fi & Fantasy"

Full example (TV Show):
    python predict.py \
        --director "Not Given" \
        --country "South Korea" \
        --date_added "1/14/2022" \
        --release_year 2021 \
        --rating "TV-MA" \
        --duration "2 Seasons" \
        --listed_in "International TV Shows, Romantic TV Shows"

Notes
-----
- Use "Not Given" for unknown director or country.
- An unseen rating or country value is handled gracefully (outputs the
  all-zero OHE row for that feature; no exception is raised).
- Run "python train.py" first to generate the model artifact.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make project root importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODELS_DIR, TARGET_MAP
from src.utils.io import load_pipeline

MODEL_PATH = MODELS_DIR / "cinevault_classifier_v1.joblib"

# Reverse label map: 1 -> Movie, 0 -> TV Show
LABEL_MAP = {v: k for k, v in TARGET_MAP.items()}


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict Netflix content type (Movie or TV Show).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--director",
        default="Not Given",
        help="Director name. Use 'Not Given' if unknown. (default: 'Not Given')",
    )
    p.add_argument(
        "--country",
        default="United States",
        help="Primary country of origin. (default: 'United States')",
    )
    p.add_argument(
        "--date_added",
        required=True,
        help="Date added to Netflix in M/D/YYYY format, e.g. '9/25/2021'.",
    )
    p.add_argument(
        "--release_year",
        type=int,
        required=True,
        help="Original release year, e.g. 2021.",
    )
    p.add_argument(
        "--rating",
        required=True,
        help="Content rating, e.g. PG-13, TV-MA, TV-PG.",
    )
    p.add_argument(
        "--duration",
        required=True,
        help="Duration string, e.g. '90 min' for a movie or '2 Seasons' for a TV show.",
    )
    p.add_argument(
        "--listed_in",
        required=True,
        help="Comma-separated genre list, e.g. 'Action & Adventure, Dramas'.",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help=f"Override the default model path (default: {MODEL_PATH}).",
    )
    return p


# ---------------------------------------------------------------------------
def run_inference(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path) if args.model_path else MODEL_PATH

    if not model_path.exists():
        print(f"[ERROR] Model artifact not found at '{model_path}'.")
        print("        Run 'python train.py' first to train and save the model.")
        sys.exit(1)

    pipeline = load_pipeline(model_path)

    # Build a single-row DataFrame with the same raw columns the model expects
    # (show_id and title are present in training data but not used by the
    # engineer; they are safe to pass as placeholder values)
    row = pd.DataFrame([{
        "show_id":      "inference",
        "title":        "inference",
        "director":     args.director,
        "country":      args.country,
        "date_added":   args.date_added,
        "release_year": args.release_year,
        "rating":       args.rating,
        "duration":     args.duration,
        "listed_in":    args.listed_in,
    }])

    pred   = pipeline.predict(row)[0]
    probs  = pipeline.predict_proba(row)[0]

    label      = LABEL_MAP[pred]
    confidence = probs[pred] * 100
    p_movie    = probs[1] * 100
    p_tvshow   = probs[0] * 100

    print()
    print(f"  Prediction   :  {label}")
    print(f"  Confidence   :  {confidence:.1f}%")
    print(f"  Probability  :  Movie = {p_movie:.1f}%  |  TV Show = {p_tvshow:.1f}%")
    print()


# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
