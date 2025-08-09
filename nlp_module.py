import os
from openai import OpenAI
from textblob import TextBlob

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_metric(document_text, question):
    """Ask OpenAI to find the answer from the document."""
    prompt = f"""
    You are a financial data extractor.
    Based only on the text below, answer the question.

    Text:
    {document_text}

    Question: {question}
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return response.output_text.strip()

def sentiment_score(document_text):
    """Use TextBlob for quick polarity score."""
    blob = TextBlob(document_text)
    return round(blob.sentiment.polarity, 3)

def detect_anomalies(current_value, previous_value, threshold=0.15):
    """Return True if change > threshold (default 15%)."""
    try:
        change = abs(current_value - previous_value) / previous_value
        return change > threshold
    except:
        return False
