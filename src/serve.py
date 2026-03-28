import glob
import os
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

# ── serve.py ───────────────────────────────────────────────────────────────────
# Serves the trained QA model via FastAPI on http://localhost:8000
#
# The /ask endpoint takes a question, searches all pages in data/pages/
# for the most relevant context, then uses the QA model to extract an answer.
#
# This is extractive QA — the model finds and returns a span of text
# from the Wikipedia pages it was trained on.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Knowledge Trainer", description="Ask the model what it learned")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_latest_model_dir() -> str:
    pattern = os.path.join(PROJECT_ROOT, "models", "round_*")
    dirs    = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    return dirs[-1]


def load_pages() -> list[dict]:
    """Load all Wikipedia pages from data/pages/"""
    pages = []
    pattern = os.path.join(PROJECT_ROOT, "data", "pages", "*.txt")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        pages.append({"title": title, "text": text, "path": filepath})
    return pages


def find_best_context(question: str, pages: list[dict], chunk_words: int = 300) -> str:
    """
    Find the most relevant context passage for a question.
    Simple keyword overlap scoring — finds the chunk with the most
    question words in it.
    """
    if not pages:
        return ""

    question_words = set(question.lower().split())
    best_chunk = ""
    best_score = -1

    for page in pages:
        words = page["text"].split()
        # Slide a window over the page
        step = chunk_words // 2
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_words])
            chunk_words_set = set(chunk.lower().split())
            score = len(question_words & chunk_words_set)
            if score > best_score:
                best_score = score
                best_chunk = chunk

    return best_chunk


# ── Load model and pages at startup ───────────────────────────────────────────
model_dir = get_latest_model_dir()
print(f"Loading model from: {model_dir}")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model     = DistilBertForQuestionAnswering.from_pretrained(model_dir)
model.eval()

pages = load_pages()
print(f"Loaded {len(pages)} Wikipedia page(s): {[p['title'] for p in pages]}")
print("Model loaded. Server ready.")


# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question:  str
    answer:    str
    context:   str
    model_dir: str


# ── /ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Answer a question using extractive QA.
    Finds the best context passage from loaded pages, then extracts the answer.
    """
    context = find_best_context(req.question, pages)

    if not context:
        return AskResponse(
            question=req.question,
            answer="I don't have any pages loaded yet. Run the pipeline first.",
            context="",
            model_dir=model_dir,
        )

    inputs = tokenizer(
        req.question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = outputs.start_logits.argmax()
    end_idx   = outputs.end_logits.argmax()

    if end_idx < start_idx:
        end_idx = start_idx

    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    if not answer:
        answer = "I couldn't find an answer in my knowledge base."

    return AskResponse(
        question=req.question,
        answer=answer,
        context=context[:300] + "...",
        model_dir=model_dir,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": model_dir,
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
