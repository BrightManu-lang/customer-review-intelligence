from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/negative_reviews.csv")
TOPIC_INFO_PATH = Path("data/topic_info.csv")
TOPIC_ASSIGNMENTS_PATH = Path("data/negative_reviews_with_topics.csv")
MODEL_DIR = Path("models/bertopic-complaints")


def cluster_complaints(sample_size: int = 3000, min_topic_size: int = 20):
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]

    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    documents = df["text"].tolist()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(documents)
    df["topic"] = topics

    topic_info = topic_model.get_topic_info()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(MODEL_DIR), serialization="safetensors")
    topic_info.to_csv(TOPIC_INFO_PATH, index=False)
    df.to_csv(TOPIC_ASSIGNMENTS_PATH, index=False)

    print(f"Saved topic info to: {TOPIC_INFO_PATH}")
    print(f"Saved assignments to: {TOPIC_ASSIGNMENTS_PATH}")
    print(topic_info.head(10))
    return topic_info


if __name__ == "__main__":
    cluster_complaints()