from preprocess_data import preprocess_reviews
from prepare_search_data import prepare_search_data
from train_transformer import train_model
from build_search_index import build_search_index
from cluster_complaints import cluster_complaints


def main():
    preprocess_reviews(sample_size=50000)
    prepare_search_data()
    train_model()
    build_search_index()
    cluster_complaints(sample_size=3000, min_topic_size=20)
    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()