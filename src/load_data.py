from pathlib import Path
import pandas as pd
import kagglehub


DATASET_HANDLE = "snap/amazon-fine-food-reviews"


def load_reviews() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = dataset_path / "Reviews.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find Reviews.csv at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def main():
    df = load_reviews()
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()