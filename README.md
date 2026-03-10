## Customer Review Intelligence Platform
A full-stack NLP application for analyzing customer feedback through sentiment classification, semantic search, complaint theme discovery, and evidence-backed summarization.

## Overview
This project turns raw customer reviews into actionable insights using modern natural language processing techniques. It combines transformer-based sentiment analysis, semantic similarity search, topic modeling, and summarization inside an interactive Gradio dashboard.

The platform is designed to help users explore customer pain points, identify recurring complaint themes, search reviews by meaning instead of exact keywords, and generate summaries supported by real review evidence.

## Features
- **Sentiment Analysis** – Classifies reviews as positive or negative using a transformer model  
- **Semantic Search** – Finds reviews similar to a query using sentence embeddings  
- **Complaint Theme Discovery** – Identifies common complaint topics using clustering  
- **Evidence-Backed Summaries** – Generates summaries supported by relevant reviews  
- **Interactive Dashboard** – Built with Gradio for easy exploration

## Tech Stack

- Python  
- Transformers  
- Sentence Transformers    
- Scikit-learn  
- BERTopic  
- Gradio  
- Matplotlib
- Pandas

## Live Demo
Hugging Face Space  
https://huggingface.co/spaces/BrightManu/customer-review-intelligence  

Sentiment Model  
https://huggingface.co/BrightManu/customer-review-sentiment  

## Running the App
Clone the repository and install dependencies.
`pip install -r requirements.txt`
`python app/app.py`

## Use Cases
- Understanding customer complaints  
- Monitoring product quality issues  
- Exploring review datasets using NLP  
- Demonstrating applied NLP techniques in an interactive app 
