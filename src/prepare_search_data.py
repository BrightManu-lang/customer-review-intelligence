from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/reviews_binary.csv")
OUTPUT_PATH = Path("data/search_reviews.csv")


def prepare_search_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    keep_cols = ["text", "summary", "Score", "HelpfulnessNumerator", "HelpfulnessDenominator", "label"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["text"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 30].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved search dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    return df


if __name__ == "__main__":
    prepare_search_data()