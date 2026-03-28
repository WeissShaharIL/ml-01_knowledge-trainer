import glob
import os
import sys
import mlflow
import mlflow.pytorch
import argparse
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch

# ── train.py ───────────────────────────────────────────────────────────────────
# Fine-tunes distilbert-base-uncased-distilled-squad on all pages in data/pages/
# using Question Answering (extractive QA).
#
# Hardware (CPU vs GPU) is controlled by config.yaml — no code changes needed.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from config import load_config, resolve_device

MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")


def load_pages(pages_dir: str):
    pattern = os.path.join(pages_dir, "*.txt")
    files   = sorted(glob.glob(pattern))
    texts   = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    return texts, files


def build_qa_dataset(texts: list[str], tokenizer, chunk_size: int = 400) -> Dataset:
    examples = []

    for text in texts:
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 20]

        for i in range(0, len(sentences) - 2, 3):
            context     = ". ".join(sentences[i:i+3]) + "."
            if len(context) < 50:
                continue

            answer_text = sentences[i][:100].strip()
            if not answer_text:
                continue

            answer_start = context.find(answer_text)
            if answer_start == -1:
                continue

            examples.append({
                "question":     "What does this text say?",
                "context":      context,
                "answer_text":  answer_text,
                "answer_start": answer_start,
            })

    if not examples:
        return None

    def tokenize_fn(example):
        encoding = tokenizer(
            example["question"],
            example["context"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        start_char  = example["answer_start"]
        end_char    = start_char + len(example["answer_text"])
        token_start = encoding.char_to_token(start_char, sequence_index=1)
        token_end   = encoding.char_to_token(end_char - 1, sequence_index=1)

        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = 0

        encoding["start_positions"] = token_start
        encoding["end_positions"]   = token_end
        return encoding

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(tokenize_fn, remove_columns=["question", "context", "answer_text", "answer_start"])
    return dataset


def main(round_num: int, epochs: int, batch_size: int):

    # ── Load config ────────────────────────────────────────────────────────────
    cfg        = load_config()
    device     = resolve_device(cfg.get("hardware", {}).get("device", "cpu"))
    MODEL_NAME = cfg.get("training", {}).get("base_model", "distilbert-base-uncased-distilled-squad")

    # Allow CLI args to override config (pipeline.py passes these)
    train_cfg  = cfg.get("training", {})
    epochs     = epochs     or train_cfg.get("epochs", 3)
    batch_size = batch_size or train_cfg.get("batch_size", 4)

    use_cpu    = (device == "cpu")

    pages_dir  = os.path.join(PROJECT_ROOT, "data", "pages")
    output_dir = os.path.join(PROJECT_ROOT, "models", f"round_{round_num}")

    texts, files = load_pages(pages_dir)

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)

    if not use_cpu:
        model = model.to(device)

    # ── Baseline mode ──────────────────────────────────────────────────────────
    if not texts:
        print("\nNo pages found in data/pages/ — saving raw pretrained baseline.")
        print("This is Round 0 — the model knows English but nothing about your topic.\n")

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        run_name = f"round-{round_num}-baseline-0pages"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("round",    round_num)
            mlflow.log_param("pages",    0)
            mlflow.log_param("model",    MODEL_NAME)
            mlflow.log_param("device",   device)
            mlflow.log_param("baseline", True)
            mlflow.log_metric("train_loss", 0)

        print(f"Baseline model saved to: {output_dir}")
        print(f"MLflow run: {run_name}")
        return

    # ── Normal training ────────────────────────────────────────────────────────
    print(f"\nRound {round_num} — training on {len(texts)} page(s)")
    for f in files:
        print(f"  {os.path.basename(f)}")

    dataset = build_qa_dataset(texts, tokenizer)
    if dataset is None or len(dataset) == 0:
        print("Could not build QA dataset from pages.")
        return

    print(f"Created {len(dataset)} QA training examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    run_name = f"round-{round_num}-{len(texts)}pages-{epochs}epochs"
    print(f"\nStarting run: {run_name}")
    print(f"Training on {device.upper()} — {'this will take a few minutes' if use_cpu else 'GPU engaged!'}...\n")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("round",      round_num)
        mlflow.log_param("pages",      len(texts))
        mlflow.log_param("examples",   len(dataset))
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model",      MODEL_NAME)
        mlflow.log_param("device",     device)

        result     = trainer.train()
        train_loss = result.training_loss
        mlflow.log_metric("train_loss", train_loss)
        mlflow.pytorch.log_model(model, name="student-model")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n── Round {round_num} complete ────────────────────────────")
        print(f"Pages        : {len(texts)}")
        print(f"Device       : {device.upper()}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Model saved  : {output_dir}")
        print(f"MLflow run   : {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int, default=0)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    main(args.round, args.epochs, args.batch_size)