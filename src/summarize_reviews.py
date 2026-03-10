from collections import Counter
import re
from semantic_search import search_reviews

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "but",
    "had", "have", "has", "too", "very", "not", "you", "your", "our", "all",
    "from", "they", "them", "their", "would", "could", "should", "there",
    "about", "into", "just", "been", "also", "than", "when", "what", "which",
    "because", "while", "after", "before", "then", "food", "product"
}


def extract_keywords(texts, top_n=8):
    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        words.extend([t for t in tokens if t not in STOPWORDS])
    return [word for word, _ in Counter(words).most_common(top_n)]


def summarize_query(query: str, top_k: int = 5):
    results = search_reviews(query, top_k=top_k)
    texts = [r["review"] for r in results]

    keywords = extract_keywords(texts, top_n=6)
    keyword_text = ", ".join(keywords) if keywords else "no dominant keywords found"

    summary = (
        f"Top retrieved reviews for '{query}' suggest recurring issues around: {keyword_text}. "
        f"Below are representative supporting reviews."
    )

    return {
        "summary": summary,
        "supporting_reviews": results
    }


if __name__ == "__main__":
    output = summarize_query("broken packaging", top_k=5)
    print(output["summary"])
    for i, item in enumerate(output["supporting_reviews"], start=1):
        print(f"{i}. {item['review'][:300]}")
        print("-" * 60)