from pathlib import Path
import sys
import re
import html

import pandas as pd
import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from semantic_search import search_reviews  # noqa: E402
from summarize_reviews import summarize_query  # noqa: E402

# SENTIMENT_MODEL_PATH = ROOT / "models" / "distilbert-review-sentiment"
TOPIC_INFO_PATH = ROOT / "data" / "topic_info.csv"
SEARCH_DATA_PATH = ROOT / "data" / "search_reviews.csv"
SEARCH_EMBEDDINGS_PATH = ROOT / "data" / "search_embeddings.npy"
DATA_PATH = ROOT / "data" / "reviews_binary.csv"

classifier = None
try:
    classifier = pipeline(
        "text-classification",
        model="BrightManu/customer-review-sentiment")
except Exception as e:
    print(f"Failed to load sentiment model: {e}")


CUSTOM_CSS = """
.gradio-container {
    max-width: 1600px !important;
    margin-left: auto;
    margin-right: auto;
}

/* Main status container */
.status-card {
    padding: 14px;
    border-radius: 14px;
    background: var(--background-fill-primary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary);
    margin-bottom: 10px;
}

/* Metric cards (Indexed Reviews, Topics, etc.) */
.metric-card {
    padding: 16px;
    border-radius: 14px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary);
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-2px);
}

/* Metric numbers */
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 4px;
    color: var(--body-text-color);
}

/* Metric labels */
.metric-label {
    font-size: 0.95rem;
    color: var(--body-text-color-subdued);
}

/* Cards used in results */
.section-card {
    padding: 14px;
    margin-bottom: 12px;
    border-radius: 14px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary);
}

/* Status badges */
.badge-ok {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #22c55e33;
    color: #16a34a;
    font-weight: 600;
}

.badge-missing {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #ef444433;
    color: #dc2626;
    font-weight: 600;
}

/* Highlighted search matches */
mark {
    background: #facc15;
    padding: 0 2px;
    border-radius: 4px;
}

/* Muted small text */
.small-muted {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
}
"""


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def get_dashboard_stats():
    review_count = 0
    topic_count = 0

    if SEARCH_DATA_PATH.exists():
        try:
            review_count = len(pd.read_csv(SEARCH_DATA_PATH))
        except Exception:
            review_count = 0

    if TOPIC_INFO_PATH.exists():
        try:
            topic_info = pd.read_csv(TOPIC_INFO_PATH)
            topic_info = topic_info[topic_info["Topic"] != -1].copy()
            topic_count = len(topic_info)
        except Exception:
            topic_count = 0

    model_ready = classifier is not None
    search_ready = SEARCH_DATA_PATH.exists() and SEARCH_EMBEDDINGS_PATH.exists()
    topics_ready = TOPIC_INFO_PATH.exists()

    return {
        "review_count": review_count,
        "topic_count": topic_count,
        "model_ready": model_ready,
        "search_ready": search_ready,
        "topics_ready": topics_ready,
    }


def status_badge(is_ready: bool):
    return (
        '<span class="badge-ok">Ready</span>'
        if is_ready
        else '<span class="badge-missing">Missing</span>'
    )


def build_status_html():
    stats = get_dashboard_stats()
    return f"""
    <div class="status-card">
        <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 10px;">
            System Status
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div class="metric-card">
                <div class="metric-label">Indexed Reviews</div>
                <div class="metric-value">{stats['review_count']:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Complaint Topics</div>
                <div class="metric-value">{stats['topic_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sentiment Model</div>
                <div class="metric-value">{'Loaded' if stats['model_ready'] else 'Not Found'}</div>
            </div>
        </div>

        <div style="margin-top: 14px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div class="section-card">
                <div><b>Sentiment Artifacts</b></div>
                <div style="margin-top: 6px;">{status_badge(stats['model_ready'])}</div>
            </div>
            <div class="section-card">
                <div><b>Semantic Search Index</b></div>
                <div style="margin-top: 6px;">{status_badge(stats['search_ready'])}</div>
            </div>
            <div class="section-card">
                <div><b>Complaint Themes</b></div>
                <div style="margin-top: 6px;">{status_badge(stats['topics_ready'])}</div>
            </div>
        </div>
    </div>
    """


def highlight_query(text: str, query: str) -> str:
    safe_text = html.escape(text)
    query_words = list(dict.fromkeys(re.findall(r"\w+", (query or "").lower())))

    for word in sorted(query_words, key=len, reverse=True):
        if len(word) < 2:
            continue
        pattern = re.compile(rf"\b({re.escape(word)})\b", flags=re.IGNORECASE)
        safe_text = pattern.sub(r"<mark>\1</mark>", safe_text)

    return safe_text


def confidence_bar(score: float, label: str) -> str:
    pct = max(0, min(100, int(round(score * 100))))
    bar_color = "#16a34a" if label == "POSITIVE" else "#dc2626"

    return f"""
    <div style="margin-top: 10px;">
        <div style="width: 100%; background: #e5e7eb; border-radius: 10px; overflow: hidden; height: 22px;">
            <div style="
                width: {pct}%;
                background: {bar_color};
                color: white;
                height: 22px;
                line-height: 22px;
                text-align: center;
                font-weight: 700;
            ">
                {pct}%
            </div>
        </div>
    </div>
    """


def shorten(text: str, max_chars: int = 450) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def predict_sentiment(review_text: str):
    review_text = (review_text or "").strip()
    if not review_text:
        return "<p>Please enter a review.</p>"

    if classifier is None:
        return "<p>Sentiment model not found. Run training first.</p>"

    result = classifier(review_text)[0]
    label = result["label"]
    confidence = float(result["score"])

    pretty_label = "Positive Review" if label == "POSITIVE" else "Negative Review"
    label_color = "#166534" if label == "POSITIVE" else "#b91c1c"

    return f"""
    <div class="section-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: {label_color};">
            {html.escape(pretty_label)}
        </div>
        <div style="margin-top: 8px;">
            Confidence: <b>{confidence:.4f}</b>
        </div>
        {confidence_bar(confidence, label)}
    </div>
    """


def search_reviews_ui(query: str, top_k: int):
    query = (query or "").strip()
    if not query:
        return "<p>Please enter a search query.</p>"

    try:
        results = search_reviews(query, top_k=int(top_k))
    except Exception as e:
        return f"<p>Search resources not available: {html.escape(str(e))}</p>"

    if not results:
        return "<p>No matching reviews found.</p>"

    cards = []
    for i, item in enumerate(results, start=1):
        review_text = shorten(item["review"], 500)
        highlighted_review = highlight_query(review_text, query)
        summary = html.escape(str(item.get("summary", "")))
        rating = item.get("rating")
        similarity = float(item["score"])

        cards.append(f"""
        <div class="section-card">
            <div style="font-weight: 700; margin-bottom: 6px;">Result {i}</div>
            <div style="margin-bottom: 6px;">
                <b>Similarity:</b> {similarity:.4f}
                &nbsp; | &nbsp;
                <b>Rating:</b> {rating}
            </div>
            <div style="margin-bottom: 6px;">
                <b>Summary:</b> {summary}
            </div>
            <div>
                <b>Review:</b> {highlighted_review}
            </div>
        </div>
        """)

    return "\n".join(cards)


def get_topic_table_html(top_n: int = 10):
    if not TOPIC_INFO_PATH.exists():
        return "<p>Run complaint clustering first.</p>"

    topic_info = pd.read_csv(TOPIC_INFO_PATH)
    topic_info = topic_info[topic_info["Topic"] != -1].copy().head(top_n)

    if topic_info.empty:
        return "<p>No complaint topics found.</p>"

    rows = []
    for _, row in topic_info.iterrows():
        topic = safe_int(row["Topic"])
        count = safe_int(row["Count"])
        name = html.escape(str(row["Name"]))
        rows.append(
            f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">{topic}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">{count}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">{name}</td>
            </tr>
            """
        )

    return f"""
    <div class="section-card">
        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 10px;">
            Top Complaint Themes
        </div>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="text-align: left; background: #f9fafb;">
                    <th style="padding: 8px; border-bottom: 1px solid #ddd;">Topic</th>
                    <th style="padding: 8px; border-bottom: 1px solid #ddd;">Count</th>
                    <th style="padding: 8px; border-bottom: 1px solid #ddd;">Theme Name</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def get_topic_chart_html(top_n: int = 8):
    if not TOPIC_INFO_PATH.exists():
        return "<p>Run complaint clustering first.</p>"

    topic_info = pd.read_csv(TOPIC_INFO_PATH)
    topic_info = topic_info[topic_info["Topic"] != -1].copy().head(top_n)

    if topic_info.empty:
        return "<p>No topic counts available.</p>"

    max_count = max(topic_info["Count"])
    bars = []

    for _, row in topic_info.iterrows():
        topic = safe_int(row["Topic"])
        count = safe_int(row["Count"])
        name = html.escape(str(row["Name"]))
        width_pct = 0 if max_count == 0 else (count / max_count) * 100

        bars.append(f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; gap: 10px; margin-bottom: 4px;">
                <div><b>Topic {topic}</b> <span class="small-muted">{name}</span></div>
                <div><b>{count}</b></div>
            </div>
            <div style="width: 100%; background: #e5e7eb; border-radius: 10px; overflow: hidden; height: 18px;">
                <div style="width: {width_pct:.1f}%; background: #2563eb; height: 18px;"></div>
            </div>
        </div>
        """)

    return f"""
    <div class="section-card">
        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 10px;">
            Complaint Topic Counts
        </div>
        {''.join(bars)}
    </div>
    """


def complaint_themes_ui():
    chart_html = get_topic_chart_html(top_n=8)
    table_html = get_topic_table_html(top_n=10)
    return chart_html + table_html


def summarize_ui(query: str, top_k: int):
    query = (query or "").strip()
    if not query:
        return "<p>Please enter a query.</p>", "<p>No supporting reviews.</p>"

    try:
        result = summarize_query(query, top_k=int(top_k))
    except Exception as e:
        msg = html.escape(str(e))
        return f"<p>Summary resources not available: {msg}</p>", "<p>No supporting reviews.</p>"

    summary_html = f"""
    <div class="section-card">
        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;">Generated Summary</div>
        <div>{html.escape(result['summary'])}</div>
    </div>
    """

    evidence_cards = []
    for i, item in enumerate(result["supporting_reviews"], start=1):
        review_text = shorten(item["review"], 450)
        highlighted_review = highlight_query(review_text, query)
        summary = html.escape(str(item.get("summary", "")))
        rating = item.get("rating")
        similarity = float(item["score"])

        evidence_cards.append(f"""
        <div class="section-card">
            <div style="font-weight: 700; margin-bottom: 6px;">Evidence {i}</div>
            <div style="margin-bottom: 6px;">
                <b>Similarity:</b> {similarity:.4f}
                &nbsp; | &nbsp;
                <b>Rating:</b> {rating}
            </div>
            <div style="margin-bottom: 6px;">
                <b>Summary:</b> {summary}
            </div>
            <div>
                <b>Review:</b> {highlighted_review}
            </div>
        </div>
        """)

    return summary_html, "\n".join(evidence_cards)


def complaint_themes_plot():
    if not TOPIC_INFO_PATH.exists():
        return None

    topic_info = pd.read_csv(TOPIC_INFO_PATH)
    topic_info = topic_info[topic_info["Topic"] != -1].copy().head(8)

    if topic_info.empty:
        return None

    labels = (
        topic_info["Name"]
        .astype(str)
        .str.replace(r"^\d+_", "", regex=True)
        .str.replace("_", " ", regex=False)
    )

    counts = topic_info["Count"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_title("Top Complaint Themes")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Theme")
    plt.tight_layout()
    return fig


def search_similarity_plot(query: str, top_k: int = 5):
    query = (query or "").strip()
    if not query:
        return None

    results = search_reviews(query, top_k=top_k)
    if not results:
        return None

    labels = [f"Result {i}" for i in range(len(results), 0, -1)]
    scores = [item["score"] for item in results][::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, scores)
    ax.set_title(f"Search Relevance: {query}")
    ax.set_xlabel("Similarity Score")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig


with gr.Blocks(title="Customer Feedback Intelligence Platform") as demo:
    gr.Markdown("# Customer Feedback Intelligence Platform")
    gr.Markdown(
        "An end-to-end NLP dashboard for sentiment classification, semantic search, complaint theme discovery, and evidence-backed summaries."
    )

    status_output = gr.HTML(value=build_status_html())

    with gr.Tab("Sentiment"):
        review_input = gr.Textbox(lines=6, placeholder="Paste a customer review...")
        gr.Examples(
            examples=[
                ["The product arrived fresh and tasted amazing. I will definitely buy it again."],
                ["The packaging was broken, the food was stale, and the smell was terrible."],
            ],
            inputs=review_input,
        )
        sentiment_output = gr.HTML(label="Prediction")
        gr.Button("Predict Sentiment").click(
            fn=predict_sentiment,
            inputs=review_input,
            outputs=sentiment_output,
        )

    with gr.Tab("Semantic Search"):
        with gr.Row():
            search_input = gr.Textbox(
                lines=2,
                placeholder="e.g. broken packaging, stale food, shipping issues",
                scale=5,
            )
            search_top_k = gr.Dropdown(
                choices=[3, 5, 10],
                value=5,
                label="Top-K Results",
                scale=1,
            )

        gr.Examples(
            examples=[
                ["broken packaging"],
                ["stale food"],
                ["shipping damage"],
                ["bad taste"],
                ["leaking containers"],
            ],
            inputs=search_input,
        )

        with gr.Row():
            search_output = gr.HTML(label="Top Matching Reviews", scale=2)
            search_plot = gr.Plot(label="Search Relevance", scale=1)

        gr.Button("Search").click(
            fn=lambda q, k: (search_reviews_ui(q, k), search_similarity_plot(q, k)),
            inputs=[search_input, search_top_k],
            outputs=[search_output, search_plot])

    with gr.Tab("Complaint Themes"):
        with gr.Row():
            themes_plot = gr.Plot(label="Top Complaint Themes", scale=1)
            themes_output = gr.HTML(label="Complaint Theme Details", scale=2)

        gr.Button("Show Themes").click(
            fn=lambda: (complaint_themes_plot(), complaint_themes_ui()),
            outputs=[themes_plot, themes_output])


    with gr.Tab("Summary with Evidence"):
        with gr.Row():
            summary_input = gr.Textbox(
                lines=2,
                placeholder="e.g. packaging complaints",
                scale=5,
            )
            summary_top_k = gr.Dropdown(
                choices=[3, 5, 10],
                value=5,
                label="Top-K Evidence",
                scale=1,
            )

        gr.Examples(
            examples=[
                ["packaging complaints"],
                ["stale food complaints"],
                ["shipping issues"],
                ["bad taste"],
            ],
            inputs=summary_input,
        )

        summary_output = gr.HTML(label="Summary")
        evidence_output = gr.HTML(label="Supporting Reviews")
        gr.Button("Generate Summary").click(
            fn=summarize_ui,
            inputs=[summary_input, summary_top_k],
            outputs=[summary_output, evidence_output],
        )

    # with gr.Tab("Analytics"):
    #     topic_plot = gr.Plot(label="Complaint Topic Distribution")
    #     sentiment_plot = gr.Plot(label="Sentiment Distribution")

    #     gr.Button("Show Analytics").click(
    #         fn=lambda: (complaint_topics_plot(), sentiment_distribution_plot()),
    #         outputs=[topic_plot, sentiment_plot])


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)