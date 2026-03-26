import os
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
import glob

# ── serve.py ───────────────────────────────────────────────────────────────────
# Phase 2 — Model serving layer.
# Starts a FastAPI server on http://localhost:8000 that loads the latest
# trained student model and answers questions via a /ask endpoint.
#
# Why FastAPI?
# It's the standard for ML model serving in Python — fast, minimal boilerplate,
# automatic API docs at /docs, and plays nicely with the transformers library.
#
# The server always loads the LATEST trained model from the models/ directory.
# When you retrain and restart the server, it automatically picks up the new
# version — no code change needed. This is the serving layer pattern used in
# production ML systems.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Knowledge Trainer", description="Ask the model what it learned")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load the latest trained model ─────────────────────────────────────────────
def get_latest_model_dir() -> str:
    """Find the most recently trained model in models/"""
    pattern = os.path.join(PROJECT_ROOT, "models", "round_*")
    dirs    = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(
            "No trained model found. Run train.py first."
        )
    return dirs[-1]  # latest round

model_dir = get_latest_model_dir()
print(f"Loading model from: {model_dir}")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model     = DistilBertForMaskedLM.from_pretrained(model_dir)
model.eval()
print("Model loaded. Server ready.")

# ── Request / response schemas ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question:    str
    answer:      str
    top_answers: list[str]
    model_dir:   str

# ── /ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Ask the student model a question.
    Uses masked language modeling — appends [MASK] and predicts the missing word.
    Returns top 5 predictions so you can see what the model is considering.
    """
    masked_input = f"Question: {req.question} Answer: [MASK]"

    inputs   = tokenizer(masked_input, return_tensors="pt", truncation=True, max_length=512)
    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits       = outputs.logits[0, mask_idx, :]
    top_ids      = logits.topk(5).indices[0].tolist()
    top_answers  = [tokenizer.decode([tid]).strip() for tid in top_ids]

    return AskResponse(
        question=req.question,
        answer=top_answers[0],
        top_answers=top_answers,
        model_dir=model_dir,
    )

# ── /health endpoint ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_dir": model_dir}

# ── /chat endpoint — serves the chat UI ───────────────────────────────────────
@app.get("/chat", response_class=HTMLResponse)
def chat():
    chat_path = os.path.join(PROJECT_ROOT, "src", "chat.html")
    with open(chat_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
