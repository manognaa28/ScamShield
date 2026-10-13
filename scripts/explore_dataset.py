from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "DataSet.csv"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== DATA TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().to_string())

    print("\n=== FRAUDULENT LABEL COUNTS ===")
    if "fraudulent" not in df.columns:
        print("ERROR: 'fraudulent' column not found.")
    else:
        counts = df["fraudulent"].value_counts(dropna=False)
        print(counts.to_string())

        if 0 in counts.index:
            print(f"\nLegitimate (0): {counts[0]:,}")
        if 1 in counts.index:
            print(f"Fraudulent (1): {counts[1]:,}")

        missing_labels = int(df["fraudulent"].isna().sum())
        print(f"Missing labels: {missing_labels:,}")


if __name__ == "__main__":
    main()
