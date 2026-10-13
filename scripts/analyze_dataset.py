from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "DataSet.csv"
STATS_PATH = ROOT / "data" / "processed" / "dataset_stats.json"

# EMSCAD stores several flags as "f" / "t" rather than 0 / 1.
LABEL_MAP = {"f": 0, "t": 1}


def json_value(value):
    """Convert pandas/NumPy values into JSON-safe Python types."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Keep DataSet.csv in the project root. This script does not rename it."
        )
    return pd.read_csv(csv_path, low_memory=False)


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map f/t flags to 0/1. Raise if the fraud label contains unexpected values."""
    if "fraudulent" not in df.columns:
        raise ValueError("The CSV has no 'fraudulent' column.")

    unexpected = sorted(
        {
            repr(value)
            for value in df["fraudulent"].dropna().unique()
            if value not in LABEL_MAP
        }
    )
    if unexpected:
        raise ValueError(
            "Unexpected values found in fraudulent labels: "
            + ", ".join(unexpected)
            + ". Expected only 'f' (legitimate) and 't' (fraudulent)."
        )

    df = df.copy()
    df["fraudulent_binary"] = df["fraudulent"].map(LABEL_MAP)

    if df["fraudulent_binary"].isna().any():
        missing = int(df["fraudulent_binary"].isna().sum())
        raise ValueError(
            f"Unexpected values found in fraudulent labels, including {missing} missing values."
        )

    df["fraudulent_binary"] = df["fraudulent_binary"].astype(int)

    for column in [
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "in_balanced_dataset",
    ]:
        df[column + "_binary"] = df[column].map(LABEL_MAP)

    return df


def fraud_rate_table(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return (
        df.groupby(group_column, dropna=False)
        .agg(
            postings=("fraudulent_binary", "size"),
            fraudulent=("fraudulent_binary", "sum"),
            fraud_rate=("fraudulent_binary", "mean"),
        )
        .reset_index()
    )


def records_from_rate_table(table: pd.DataFrame, group_column: str, label_name: str, label_fn):
    records = []
    for row in table.itertuples(index=False):
        group_value = getattr(row, group_column)
        records.append(
            {
                label_name: label_fn(group_value),
                "postings": int(row.postings),
                "fraudulent_postings": int(row.fraudulent),
                "fraud_rate": float(row.fraud_rate),
            }
        )
    return records


def logo_label(value) -> str:
    if value == 1:
        return "has_logo"
    if value == 0:
        return "no_logo"
    return "unknown"


def profile_label(value) -> str:
    return "has_company_profile" if bool(value) else "missing_company_profile"


def length_stats(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()) if len(series) else None,
        "median": float(series.median()) if len(series) else None,
        "min": int(series.min()) if len(series) else None,
        "max": int(series.max()) if len(series) else None,
    }


def build_stats(df: pd.DataFrame) -> dict:
    total_postings = int(len(df))
    fraudulent_postings = int(df["fraudulent_binary"].sum())
    legitimate_postings = total_postings - fraudulent_postings

    logo_table = fraud_rate_table(df, "has_company_logo_binary")
    profile_table = fraud_rate_table(df, "has_company_profile")
    employment_table = (
        fraud_rate_table(df, "employment_type_clean")
        .sort_values("postings", ascending=False)
        .reset_index(drop=True)
    )

    legitimate_lengths = df.loc[df["fraudulent_binary"] == 0, "description_length"]
    fraudulent_lengths = df.loc[df["fraudulent_binary"] == 1, "description_length"]

    crosstab = pd.crosstab(
        df["in_balanced_dataset"],
        df["fraudulent"],
        dropna=False,
    )
    crosstab_records = []
    for balanced_value, row in crosstab.iterrows():
        for fraud_value, count in row.items():
            crosstab_records.append(
                {
                    "in_balanced_dataset": json_value(balanced_value),
                    "fraudulent": json_value(fraud_value),
                    "postings": int(count),
                }
            )

    return {
        "source_file": str(DATA_PATH.name),
        "dataset_totals": {
            "total_postings": total_postings,
            "legitimate_postings": legitimate_postings,
            "fraudulent_postings": fraudulent_postings,
            "fraud_rate": float(df["fraudulent_binary"].mean()),
        },
        "fraud_rate_by_company_logo": records_from_rate_table(
            logo_table,
            "has_company_logo_binary",
            "company_logo",
            logo_label,
        ),
        "fraud_rate_by_company_profile": records_from_rate_table(
            profile_table,
            "has_company_profile",
            "company_profile",
            profile_label,
        ),
        "fraud_rate_by_employment_type": records_from_rate_table(
            employment_table,
            "employment_type_clean",
            "employment_type",
            lambda value: "Unknown" if pd.isna(value) else str(value),
        ),
        "description_length_by_fraud_label": {
            "legitimate": length_stats(legitimate_lengths),
            "fraudulent": length_stats(fraudulent_lengths),
        },
        "in_balanced_dataset_crosstab": {
            "purpose": (
                "Data-quality documentation only. "
                "in_balanced_dataset is not a predictive feature and must not be used for training."
            ),
            "counts": crosstab_records,
        },
    }


def write_stats(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    df = load_dataset(DATA_PATH)
    df = add_binary_labels(df)

    # Calculate the length of the job description.
    df["description_length"] = df["description"].fillna("").str.len()
    df["has_company_profile"] = (
        df["company_profile"].fillna("").str.strip().ne("")
    )
    df["employment_type_clean"] = (
        df["employment_type"].fillna("Unknown").replace("", "Unknown")
    )

    print("\n=== OVERALL FRAUD RATE ===")
    print(f"Total postings: {len(df):,}")
    print(f"Fraudulent postings: {df['fraudulent_binary'].sum():,}")
    print(f"Fraud rate: {df['fraudulent_binary'].mean():.2%}")

    print("\n=== FRAUD RATE BY COMPANY LOGO ===")
    print(
        df.groupby("has_company_logo_binary", dropna=False)
        .agg(
            postings=("fraudulent_binary", "size"),
            fraudulent=("fraudulent_binary", "sum"),
            fraud_rate=("fraudulent_binary", "mean"),
        )
        .to_string()
    )

    print("\n=== FRAUD RATE BY COMPANY PROFILE AVAILABILITY ===")
    print(
        df.groupby("has_company_profile")
        .agg(
            postings=("fraudulent_binary", "size"),
            fraudulent=("fraudulent_binary", "sum"),
            fraud_rate=("fraudulent_binary", "mean"),
        )
        .to_string()
    )

    print("\n=== FRAUD RATE BY EMPLOYMENT TYPE ===")
    print(
        df.groupby("employment_type_clean")
        .agg(
            postings=("fraudulent_binary", "size"),
            fraudulent=("fraudulent_binary", "sum"),
            fraud_rate=("fraudulent_binary", "mean"),
        )
        .sort_values("postings", ascending=False)
        .to_string()
    )

    print("\n=== DESCRIPTION LENGTH BY FRAUD LABEL ===")
    print(
        df.groupby("fraudulent_binary")["description_length"]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string()
    )

    print("\n=== BALANCED-DATASET FLAG VS FRAUD LABEL ===")
    print(
        pd.crosstab(
            df["in_balanced_dataset"],
            df["fraudulent"],
            margins=True,
            dropna=False,
        ).to_string()
    )

    print("\n=== LOCATION SUMMARY ===")
    # The dataset's location field is not guaranteed to contain a
    # consistently formatted country. Show frequent raw values first.
    print(df["location"].fillna("Unknown").value_counts().head(20).to_string())

    stats = build_stats(df)
    write_stats(stats, STATS_PATH)

    print("\nDay 2 analysis completed.")
    print(f"Wrote computed statistics to: {STATS_PATH}")
    print("DataSet.csv was not modified.")
    print("No model was trained.")


if __name__ == "__main__":
    main()
