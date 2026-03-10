from pathlib import Path
import pandas as pd
from load_data import load_reviews

OUTPUT_PATH = Path("data/reviews_binary.csv")
NEGATIVE_OUTPUT_PATH = Path("data/negative_reviews.csv")


def score_to_label(score: int):
    if score in [1, 2]:
        return 0
    if score in [4, 5]:
        return 1
    return None


def preprocess_reviews(sample_size: int = 50000):

    df = load_reviews()

    keep_cols = ["Text", "Summary", "Score", "HelpfulnessNumerator", "HelpfulnessDenominator"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["Text", "Score"]).copy()

    df["label"] = df["Score"].apply(score_to_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df = df.rename(columns={"Text": "text", "Summary": "summary"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20].reset_index(drop=True)

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    negative_df = df[df["label"] == 0].copy()
    negative_df.to_csv(NEGATIVE_OUTPUT_PATH, index=False)

    print(f"Saved processed data to: {OUTPUT_PATH}")
    print(f"Saved negative reviews to: {NEGATIVE_OUTPUT_PATH}")
    print(df["label"].value_counts())

    return df


if __name__ == "__main__":
    preprocess_reviews()