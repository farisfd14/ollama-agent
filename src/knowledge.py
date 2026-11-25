# src/knowledge.py
# Milestone 2: turn PDFs into a searchable index.
#
# The RAG idea in four moves:
#   1. EXTRACT  - pull raw text out of every PDF in data/
#   2. CHUNK    - cut text into small pieces (~300 chars) because a
#                 whole page is too big to match a question precisely
#   3. EMBED    - convert each chunk into a vector of numbers that
#                 captures its MEANING (this is the expensive step,
#                 so we cache it to disk - same lesson as stock-lab:
#                 never redo slow work)
#   4. RETRIEVE - embed the question the same way, then return the
#                 chunks whose vectors are closest (cosine similarity)

import json
import os

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from torch.distributed._shard.sharding_spec import chunk_sharding_spec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.json")

# multilingual = understands English AND Chinese notes
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 50      # chunks share a little text so ideas at the
                        # boundary don't get cut in half


def load_pdf_texts():
    """Read every PDF in data/ -> list of (source_name, full_text)."""
    pages = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(DATA_DIR, fname))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            pages.append((fname, text))
    return pages


def chunk_text(text, source):
    """Cut one document into overlapping chunks.
    Returns a list of dicts: {text, source}."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        if piece.strip():
            chunks.append({"text": piece.strip(), "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    """Embed all chunks and save them to data/index.json (one time)."""
    model = SentenceTransformer(EMBED_MODEL)
    pages = load_pdf_texts()
    if not pages:
        raise RuntimeError("no PDFs found in data/ folder")

    chunks = []
    for source, text in pages:
        chunks.extend(chunk_text(text, source))

    print(f"embedding {len(chunks)} chunks...")
    vectors = model.encode([c["text"] for c in chunks],
                           normalize_embeddings=True)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks,
                   "vectors": vectors.tolist()}, f)
    print(f"saved {INDEX_PATH}")


def search(query, top_k=3):
    """Return the `top_k` chunks most similar to the question."""
    model = SentenceTransformer(EMBED_MODEL)
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    q_vec = model.encode([query], normalize_embeddings=True)[0]
    matrix = np.array(index["vectors"])
    # cosine similarity: 1.0 = same direction, 0 = unrelated
    scores = matrix @ q_vec

    best = np.argsort(scores)[::-1][:top_k]
    return [(index["chunks"][i], float(scores[i])) for i in best]


if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        build_index()
    # manual test: py src/knowledge.py "your question here"
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "what is this document about?"
    for chunk, score in search(question):
        print(f"\n[{score:.3f}] ({chunk['source']})")
        print(chunk["text"][:200])
