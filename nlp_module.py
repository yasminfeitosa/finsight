from transformers import pipeline
from textblob import TextBlob
import re

# Load QA model once
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_metric(document_text, question):
    """Extract numeric or textual answer from document."""
    answer = qa_pipeline(question=question, context=document_text)
    return answer.get("answer", "N/A")

def sentiment_score(document_text):
    """Quick sentiment polarity score (-1 to +1)."""
    blob = TextBlob(document_text)
    return round(blob.sentiment.polarity, 3)

def detect_anomalies(current_value, previous_value, threshold=0.15):
    """Return True if change > threshold (default 15%)."""
    try:
        change = abs(current_value - previous_value) / previous_value
        return change > threshold
    except:
        return False