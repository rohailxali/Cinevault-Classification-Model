# Cinevault Classification Model

This project builds a professional, production-quality machine learning system to classify titles as **Movie** or **TV Show** using tabular metadata. The dataset is a metadata table of 8,790 titles.

## Architecture

The project is structured into clear responsibilities:
```
Project 2/
├── data/
│   ├── raw/Dataset.csv       # Original data (read-only)
│   └── processed/            # Ephemeral processed files (if any)
├── models/                   # Saved model artifacts (.joblib)
├── reports/                  # Evaluation metrics output
├── src/
│   ├── config.py             # Global constants (seed, paths)
│   ├── data/loader.py        # Loading and validation
│   ├── evaluation/metrics.py # Reporting logic
│   ├── features/engineer.py  # Feature engineering transformer
│   ├── models/trainer.py     # ML Pipeline construction
│   └── utils/io.py           # Persistence wrappers
├── tests/                    # pytest suite
├── train.py                  # Single-entry training script
└── predict.py                # Standalone inference CLI
```

## Dataset Analysis & Challenges

An initial inspection revealed a few properties critical to this pipeline:
- **Rows/Cols**: 8,790 rows x 10 columns.
- **Nulls**: There were zero true `NaN` values; missing data was encoded as the string `"Not Given"`.
- **Target Balance**: 69.7% Movie, 30.3% TV Show. Since the classes are imbalanced, Accuracy is an unreliable metric, and `macro F1` was chosen as the primary metric, with `ROC-AUC` as a tiebreaker.
- **Duration Leakage**: The dataset has a `duration` column encoded as `"X min"` for Movies and `"X Season(s)"` for TV Shows. This means the unit alone perfectly predicts the target variable. 
  - To handle this, the model uses a dual-configuration approach. **Config A** includes the `duration_is_minutes` flag to demonstrate the near-100% artificial ceiling. **Config B** fully excludes the `duration` feature to conduct a genuine, rigorous model comparison based on the remaining difficult signal.

## Modeling & Pipeline

1. **Feature Engineering**: 
   - Extract primary genre and total genre count from `listed_in`.
   - Extract primary country and group rare countries into `"Other"`.
   - Create a boolean indicator `has_director` to replace the high-cardinality `director` field which is missing 30% of the time.
   - Extract the year and month added from `date_added`, and derive `years_since_release`.
   - Group thin-tail values in `rating` into `"Rare"`.
2. **Preprocessing**: Handled inside an `sklearn.compose.ColumnTransformer`.
3. **Training Models**:
   - DummyClassifier (floor baseline)
   - LogisticRegression
   - DecisionTree
   - RandomForest (with GridSearch)
4. **Validation**: 5-fold stratified Cross-Validation (on training data only).
5. **Selection criterion**: Highest `macro F1-score` on CV (training data only). 

## Evaluation Results

The final winning model for **Config B** (no duration leak) was **LogisticRegression**.

| Model | CV_F1_macro | Test_Accuracy | Test_F1_macro | Test_ROC_AUC | F1_Movie | F1_TVShow |
| --- | --- | --- | --- | --- | --- | --- |
| DummyClassifier | 0.4107 | 0.6968 | 0.4107 | 0.5000 | 0.8213 | 0.0000 |
| **LogisticRegression (Winner)** | **0.9971** | **0.9977** | **0.9973** | **0.9999** | **0.9984** | **0.9962** |
| DecisionTree | 0.9951 | 0.9966 | 0.9960 | 0.9960 | 0.9976 | 0.9944 |
| RandomForest | 0.9958 | 0.9966 | 0.9960 | 0.9999 | 0.9976 | 0.9944 |

## Reproducibility & Setup

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model and reproduce the report:**
   ```bash
   python train.py
   ```
   *The script generates evaluation reports in `reports/` and saves the winning pipeline to `models/cinevault_classifier_v1.joblib`.*

3. **Run standalone inference:**
   ```bash
   python predict.py \
       --director "Christopher Nolan" \
       --country "United States" \
       --date_added "7/16/2010" \
       --release_year 2010 \
       --rating "PG-13" \
       --duration "148 min" \
       --listed_in "Action & Adventure, Sci-Fi & Fantasy"
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```
