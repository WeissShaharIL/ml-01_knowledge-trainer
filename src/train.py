import glob
import os
import mlflow
import mlflow.pytorch
import argparse
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# ── train.py ───────────────────────────────────────────────────────────────────
# Step 3 of the knowledge trainer pipeline.
# Fine-tunes distilbert-base-uncased on all pages in data/pages/ using
# Masked Language Modeling (MLM) — the same technique used to pretrain BERT.
#
# What is MLM?
# The model reads your text with some words randomly masked out (like a cloze
# test), and learns to predict the missing words. This forces it to build a
# deep understanding of your domain's vocabulary, facts, and relationships.
# It's not learning to answer questions directly — it's internalizing the text
# so deeply that it can later retrieve facts from it.
#
# Why MLM instead of supervised fine-tuning?
# Supervised fine-tuning needs labeled (question, answer) pairs for EVERY fact
# you want the model to know. MLM needs only raw text — which is exactly what
# Wikipedia gives us. Much simpler data pipeline, same end result.
#
# Every run is tracked in MLflow so you can compare rounds side by side.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")

MODEL_NAME = "distilbert-base-uncased"


def load_pages(pages_dir: str) -> list[str]:
    """Load all .txt files from pages_dir and return as a list of strings."""
    pattern = os.path.join(pages_dir, "*.txt")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No .txt files found in {pages_dir}")

    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())

    print(f"Loaded {len(files)} page(s) from {pages_dir}/")
    return texts


def chunk_texts(texts: list[str], tokenizer, chunk_size: int = 512) -> Dataset:
    """
    Split texts into chunks of chunk_size tokens.
    Transformers have a max input length (512 tokens for distilbert).
    Long Wikipedia articles need to be split into overlapping windows
    so no information is lost at chunk boundaries.
    """
    all_chunks = []
    for text in texts:
        tokens = tokenizer(
            text,
            truncation=False,
            return_tensors=None,
        )["input_ids"]

        # Slide a window of chunk_size tokens with 50-token overlap
        step = chunk_size - 50
        for i in range(0, len(tokens), step):
            chunk = tokens[i : i + chunk_size]
            if len(chunk) > 32:  # skip very short trailing chunks
                all_chunks.append({"input_ids": chunk})

    print(f"Created {len(all_chunks)} text chunks for training")
    return Dataset.from_list(all_chunks)


def main(round_num: int, epochs: int, batch_size: int):

    pages_dir = os.path.join(PROJECT_ROOT, "data", "pages")
    texts     = load_pages(pages_dir)

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForMaskedLM.from_pretrained(MODEL_NAME)

    dataset = chunk_texts(texts, tokenizer)

    # MLM data collator — randomly masks 15% of tokens during training.
    # 15% is the value used in the original BERT paper and works well in practice.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    output_dir = os.path.join(PROJECT_ROOT, "models", f"round_{round_num}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",         # we let MLflow handle artifact storage
        logging_steps=10,
        report_to="none",           # disable default logging, we use MLflow
        no_cuda=True,               # CPU training — set to False if you have a GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    run_name = f"round-{round_num}-{len(texts)}pages-{epochs}epochs"
    print(f"\nStarting training run: {run_name}")
    print(f"Pages: {len(texts)} · Chunks: {len(dataset)} · Epochs: {epochs}")
    print("This will take a few minutes on CPU...\n")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("round",       round_num)
        mlflow.log_param("pages",       len(texts))
        mlflow.log_param("chunks",      len(dataset))
        mlflow.log_param("epochs",      epochs)
        mlflow.log_param("batch_size",  batch_size)
        mlflow.log_param("model",       MODEL_NAME)
        mlflow.log_param("mlm_prob",    0.15)

        result = trainer.train()

        train_loss = result.training_loss
        mlflow.log_metric("train_loss", train_loss)

        # Save model artifact to MLflow
        mlflow.pytorch.log_model(model, name="student-model")

        # Also save locally for evaluate.py to load quickly
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_param("model_dir", output_dir)

        print(f"\n── Round {round_num} complete ─────────────────────────")
        print(f"Pages trained on : {len(texts)}")
        print(f"Training loss    : {train_loss:.4f}")
        print(f"Model saved to   : {output_dir}")
        print(f"MLflow run       : {run_name}")
        print(f"\nNext step: run evaluate.py --model-dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int, default=1)
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Training epochs (more = better but slower)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args.round, args.epochs, args.batch_size)
