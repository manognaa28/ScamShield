"""
Train a baseline job-ad risk model: TF-IDF + logistic regression.

The split happens before any fitting so TF-IDF cannot learn from the test set.
in_balanced_dataset is never used as a feature.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "DataSet.csv"
MODEL_PATH = ROOT / "artifacts" / "scamshield_model.joblib"
EVAL_PATH = ROOT / "data" / "processed" / "model_evaluation.json"

# f = legitimate, t = fraudulent (EMSCAD encoding in this CSV).
LABEL_MAP = {"f": 0, "t": 1}
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
REQUIRED_COLUMNS = TEXT_COLUMNS + ["fraudulent"]
# Listed so it is obvious this column is excluded from X.
EXCLUDED_COLUMNS = ["in_balanced_dataset"]


def json_number(value):
    """Turn NumPy scalars into plain Python numbers for JSON."""
    if hasattr(value, "item"):
        return value.item()
    return value


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Keep DataSet.csv in the project root. This script does not modify it."
        )
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except OSError as error:
        raise OSError(f"Could not read the dataset at {csv_path}: {error}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse {csv_path} as CSV: {error}") from error


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    if df["fraudulent"].isna().any():
        missing_labels = int(df["fraudulent"].isna().sum())
        raise ValueError(
            f"The fraudulent column has {missing_labels} missing label(s). "
            "Every row must have a target value before training."
        )

    unexpected = sorted(
        {
            repr(value)
            for value in df["fraudulent"].unique()
            if value not in LABEL_MAP
        }
    )
    if unexpected:
        raise ValueError(
            "Unexpected values in fraudulent labels: "
            + ", ".join(unexpected)
            + ". Expected only 'f' (legitimate) and 't' (fraudulent)."
        )


def prepare_text_and_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Combine text fields into one document per posting. Do not use excluded columns."""
    text_frame = df[TEXT_COLUMNS].fillna("").astype(str)
    X = text_frame.agg(" ".join, axis=1)
    y = df["fraudulent"].map(LABEL_MAP).astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    # Vectorizer and classifier live in one Pipeline so transform() on new
    # text always uses the vocabulary learned from the training split only.
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_on_test(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    # Probability of class 1 (fraudulent) is used for average precision.
    y_score = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Classification report (test set) ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("=== Confusion matrix (labels [0, 1]) ===")
    print(matrix)
    print("Rows = actual class, columns = predicted class.")
    print("[[true legitimate, false fraudulent],")
    print(" [false legitimate, true fraudulent]]")

    accuracy = accuracy_score(y_test, y_pred)
    fraud_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    fraud_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    fraud_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    avg_precision = average_precision_score(y_test, y_score)

    print("\n=== Fraudulent-class metrics (test set) ===")
    print(f"Accuracy (overall): {accuracy:.4f}")
    print(f"Precision (fraudulent): {fraud_precision:.4f}")
    print(f"Recall (fraudulent): {fraud_recall:.4f}")
    print(f"F1-score (fraudulent): {fraud_f1:.4f}")
    print(f"Average precision (fraudulent ranking): {avg_precision:.4f}")

    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy),
        "fraudulent_precision": float(fraud_precision),
        "fraudulent_recall": float(fraud_recall),
        "fraudulent_f1": float(fraud_f1),
        "average_precision": float(avg_precision),
        "confusion_matrix": {
            "labels": [0, 1],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": matrix.tolist(),
        },
    }


def count_labels(y: pd.Series) -> dict:
    return {
        "legitimate": int((y == 0).sum()),
        "fraudulent": int((y == 1).sum()),
        "total": int(len(y)),
    }


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(pipeline, path)
    except OSError as error:
        raise OSError(f"Could not save the model to {path}: {error}") from error


def save_evaluation(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(f"Could not save evaluation metrics to {path}: {error}") from error


def main() -> None:
    print(f"Loading dataset: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    validate_dataset(df)

    if "in_balanced_dataset" in df.columns:
        print(
            "Note: in_balanced_dataset is present in the CSV but is not used "
            "as a training feature."
        )

    X, y = prepare_text_and_labels(df)

    # Split first, then fit. Stratify keeps the fraud rate similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Train rows: {len(X_train):,} | Fraudulent in train: {(y_train == 1).sum():,}")
    print(f"Test rows: {len(X_test):,} | Fraudulent in test: {(y_test == 1).sum():,}")

    pipeline = build_pipeline()
    print("\nFitting TF-IDF and logistic regression on the training split only...")
    pipeline.fit(X_train, y_train)

    metrics = evaluate_on_test(pipeline, X_test, y_test)

    evaluation = {
        "model": {
            "type": "sklearn Pipeline",
            "steps": ["TfidfVectorizer", "LogisticRegression"],
            "class_weight": "balanced",
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "text_columns": TEXT_COLUMNS,
            "excluded_columns": EXCLUDED_COLUMNS,
        },
        "dataset": {
            "source_file": DATA_PATH.name,
            "row_count": json_number(len(df)),
            "label_counts": count_labels(y),
        },
        "split": {
            "train_row_count": json_number(len(X_train)),
            "test_row_count": json_number(len(X_test)),
            "train_label_counts": count_labels(y_train),
            "test_label_counts": count_labels(y_test),
        },
        "test_metrics": metrics,
        "disclaimer": (
            "These scores are estimates from a held-out portion of historical "
            "EMSCAD job ads. They are not a guarantee that a new posting is or "
            "is not fraudulent."
        ),
    }

    save_pipeline(pipeline, MODEL_PATH)
    save_evaluation(evaluation, EVAL_PATH)

    print(f"\nSaved trained pipeline to: {MODEL_PATH}")
    print(f"Saved evaluation metrics to: {EVAL_PATH}")
    print("DataSet.csv and dataset_stats.json were not modified.")


if __name__ == "__main__":
    main()
