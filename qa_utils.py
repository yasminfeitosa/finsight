import re
import time
import openai
import streamlit as st

def choose_best_chunk(question: str, chunks: list) -> tuple[str, int]:
    """
    Find the chunk with the highest word overlap with the question.

    Args:
        question: The user's question.
        chunks: List of text chunks.

    Returns:
        Tuple of (best_chunk, index).
    """
    if not chunks:
        return "", 0
    q_words = set(re.findall(r"\w+", question.lower()))
    best_idx, best_score = 0, -1
    for i, c in enumerate(chunks):
        c_words = set(re.findall(r"\w+", c.lower()))
        score = len(q_words & c_words)
        if score > best_score:
            best_score = score
            best_idx = i
    return chunks[best_idx], best_idx

def openai_answer_with_context(question: str, chunks: list, n_chunks: int = 2, model: str = "gpt-4o-mini", max_retries: int = 3) -> tuple[str, str]:
    """
    Uses OpenAI GPT to answer a question based on relevant document chunks.
    Includes retry logic on failures.

    Args:
        question: User's question string.
        chunks: List of text chunks.
        n_chunks: Number of chunks before and after best chunk to include.
        model: OpenAI model name.
        max_retries: Number of retry attempts on failure.

    Returns:
        Tuple of (answer text, snippet used).
    """
    if not openai.api_key:
        return "OpenAI key missing: install in secrets or input in UI", "No snippet"
    if not chunks:
        return "No document uploaded", "N/A"

    main_chunk, idx = choose_best_chunk(question, chunks)

    # Select surrounding chunks for context
    selected = [main_chunk]
    for k in range(1, n_chunks):
        if idx - k >= 0:
            selected.append(chunks[idx - k])
        if idx + k < len(chunks):
            selected.append(chunks[idx + k])

    context = "\n\n---\n\n".join(selected)

    system_prompt = (
        "You are a precise financial analyst. Answer based ONLY on the provided context. "
        "Cite the context by returning a short snippet identifier (e.g., 'Source snippet: first 120 chars'). "
        "If the answer is not present, say 'NOT FOUND in context'. Keep answers concise."
    )
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer concisely and include 'Source snippet:' followed by a short snippet from context if used."
    )

    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.0
            )
            text = resp["choices"][0]["message"]["content"].strip()
            m = re.search(r"Source snippet:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            snippet = m.group(1).strip()[:400] if m else (main_chunk[:200].replace("\n", " "))
            return text, snippet
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)  # Exponential backoff
                continue
            return f"OpenAI error: {e}", main_chunk[:200].replace("\n", " ")

def perform_qa(text: str, question: str) -> tuple[str, str]:
    """
    Simple keyword-based heuristic Q&A fallback.

    Args:
        text: Full document text.
        question: User question.

    Returns:
        Tuple of (answer string, snippet).
    """
    keywords = re.findall(r"\w+", question.lower())
    sentences = re.split(r"(?<=[.!?]) +", text)
    best_sent = ""
    best_count = 0
    for s in sentences:
        s_lower = s.lower()
        count = sum(1 for kw in keywords if kw in s_lower)
        if count > best_count:
            best_count = count
            best_sent = s
    if best_sent:
        snippet = best_sent[:200].replace("\n", " ")
        answer = best_sent.strip()
        return answer, snippet
    else:
        return "NOT FOUND in document", ""
