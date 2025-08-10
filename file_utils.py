import re
import PyPDF2
import streamlit as st

def read_uploaded_file(file) -> str:
    """
    Reads text content from an uploaded file.
    Supports PDF and TXT files.

    Args:
        file: Uploaded file object from Streamlit file_uploader

    Returns:
        Extracted text as a string, or empty string on failure.
    """
    if file is None:
        return ""
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        elif name.endswith(".txt"):
            raw = file.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        else:
            st.warning(f"Unsupported file type: {name}")
            return ""
    except Exception as e:
        st.warning(f"Could not read file '{name}': {e}")
        return ""

def chunk_text_semantic(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Splits text into chunks roughly max_chunk_size in length,
    but tries to break on paragraph or sentence boundaries for better semantic coherence.

    Args:
        text: The full text string to chunk.
        max_chunk_size: Max chars per chunk.
        overlap: Number of chars to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If paragraph itself too big, split by sentences:
            if len(para) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?]) +', para)
                sent_chunk = ""
                for sent in sentences:
                    if len(sent_chunk) + len(sent) + 1 <= max_chunk_size:
                        sent_chunk += sent + " "
                    else:
                        if sent_chunk:
                            chunks.append(sent_chunk.strip())
                        sent_chunk = sent + " "
                if sent_chunk:
                    chunks.append(sent_chunk.strip())
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap by merging chunks partially if needed (optional)
    # For simplicity, skipping overlap in this semantic approach.

    return chunks

def chunk_text_fixed(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Original fixed-length chunking with overlap.

    Args:
        text: Input text string.
        chunk_size: Number of chars per chunk.
        overlap: Number of chars to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
