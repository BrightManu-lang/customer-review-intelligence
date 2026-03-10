from transformers import pipeline

MODEL_PATH = "models/distilbert-review-sentiment"


def test_predictions():
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

    examples = [
        "This product tastes amazing and I will buy it again.",
        "The package arrived broken and the food was stale.",
    ]

    for text in examples:
        result = classifier(text)[0]
        print(f"\nReview: {text}")
        print(f"Prediction: {result['label']} | Confidence: {result['score']:.4f}")


if __name__ == "__main__":
    test_predictions()