from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/search_reviews.csv")
EMBEDDINGS_PATH = Path("data/search_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_df = None
_embeddings = None


def _load_resources():
    global _model, _df, _embeddings

    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_PATH)
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model, _df, _embeddings


def search_reviews(query: str, top_k: int = 5):
    model, df, embeddings = _load_resources()

    query = (query or "").strip()
    if not query:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "score": float(scores[idx]),
            "summary": str(row.get("summary", "")),
            "review": str(row["text"]),
            "rating": int(row["Score"]) if "Score" in row and not pd.isna(row["Score"]) else None,
        })
    return results


if __name__ == "__main__":
    results = search_reviews("broken packaging and leaking containers", top_k=5)
    for i, item in enumerate(results, start=1):
        print(f"{i}. similarity={item['score']:.4f}")
        print(f"rating={item['rating']}")
        print(f"summary={item['summary']}")
        print(f"review={item['review'][:400]}")
        print("-" * 80)