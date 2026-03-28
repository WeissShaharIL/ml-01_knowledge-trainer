import glob
import os
import re
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ── serve.py ───────────────────────────────────────────────────────────────────
# RAG pipeline — BM25 retrieval + Ollama generative reader.
#
# For each question:
#   1. BM25 finds the top 3 most relevant chunks from all ingested pages
#   2. Ollama reads those chunks and generates a natural language answer
#
# No DistilBERT, no span extraction — fully generative.

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Knowledge Trainer", description="Ask the model what it learned")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_pages() -> list[dict]:
    """Load all pages from data/pages/."""
    pages   = []
    pattern = os.path.join(PROJECT_ROOT, "data", "pages", "*.txt")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        pages.append({"title": title, "text": text, "path": filepath})
    return pages


def find_best_context(question: str, pages: list[dict], chunk_words: int = 300) -> str:
    """BM25 retrieval — returns top 3 chunks concatenated."""
    if not pages:
        return ""

    stop_words = {"what", "is", "the", "a", "an", "of", "in", "was", "were",
                  "how", "when", "where", "who", "why", "did", "do", "does",
                  "that", "this", "which", "by", "at", "from", "with", "and",
                  "or", "to", "for", "on", "are", "has", "had", "have", "be"}

    chunks = []
    for page in pages:
        words = page["text"].split()
        step  = chunk_words // 2
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + chunk_words]))

    if not chunks:
        return ""

    tokenized    = [[w for w in c.lower().split() if w not in stop_words] for c in chunks]
    query_tokens = [w for w in question.lower().split() if w not in stop_words]

    if not query_tokens:
        return chunks[0]

    bm25        = BM25Okapi(tokenized)
    scores      = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[::-1][:3]
    return " ".join(chunks[i] for i in top_indices)


def clean_context(context: str) -> str:
    """Strip TITLE headers and === separators."""
    context = re.sub(r'TITLE:.*?(?=\w{4})', '', context, flags=re.DOTALL)
    context = re.sub(r'=+\s*', ' ', context).strip()
    return context


def call_ollama(question: str, context: str) -> str:
    """Ask Ollama to generate an answer from the retrieved context."""
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have information about that."
Keep your answer concise — 1 to 3 sentences maximum.

Context:
{context}

Question: {question}

Answer:"""

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


# ── Load pages at startup ──────────────────────────────────────────────────────
pages = load_pages()
print(f"Loaded {len(pages)} page(s): {[p['title'] for p in pages]}")
print("Server ready.")


# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer:   str
    context:  str


# ── /ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    context = find_best_context(req.question, pages)

    if not context:
        return AskResponse(
            question=req.question,
            answer="No pages loaded. Run the pipeline first.",
            context="",
        )

    context_clean = clean_context(context)
    answer        = call_ollama(req.question, context_clean)

    print(f"[ASK] Q: {req.question!r}")
    print(f"[ASK] A: {answer!r}")

    return AskResponse(
        question=req.question,
        answer=answer,
        context=context_clean[:300] + "...",
    )


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "mode":         "rag-generative",
        "retrieval":    "bm25",
        "reader":       OLLAMA_MODEL,
        "pages_loaded": [p["title"] for p in pages],
    }


@app.get("/chat", response_class=HTMLResponse)
def chat():
    chat_path = os.path.join(PROJECT_ROOT, "src", "chat.html")
    with open(chat_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)