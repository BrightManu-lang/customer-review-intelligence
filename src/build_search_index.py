from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/search_reviews.csv")
EMBEDDINGS_PATH = Path("data/search_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_search_index():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].fillna("").astype(str).tolist()

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding reviews...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    build_search_index()