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
# Fine-tunes distilbert-base-uncased on all pages in data/pages/ using
# Masked Language Modeling (MLM).
#
# If data/pages/ is empty (round 0 baseline), we skip fine-tuning entirely
# and just save the raw pretrained distilbert. This gives us a clean baseline
# to compare against — the model before it learned anything from our data.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")

MODEL_NAME = "distilbert-base-uncased"


def load_pages(pages_dir: str) -> list[str]:
    pattern = os.path.join(pages_dir, "*.txt")
    files   = sorted(glob.glob(pattern))
    texts   = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    return texts, files


def chunk_texts(texts: list[str], tokenizer, chunk_size: int = 512) -> Dataset:
    all_chunks = []
    for text in texts:
        tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        step   = chunk_size - 50
        for i in range(0, len(tokens), step):
            chunk = tokens[i : i + chunk_size]
            if len(chunk) > 32:
                all_chunks.append({"input_ids": chunk})
    return Dataset.from_list(all_chunks)


def main(round_num: int, epochs: int, batch_size: int):

    pages_dir  = os.path.join(PROJECT_ROOT, "data", "pages")
    output_dir = os.path.join(PROJECT_ROOT, "models", f"round_{round_num}")

    texts, files = load_pages(pages_dir)

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForMaskedLM.from_pretrained(MODEL_NAME)

    # ── Baseline mode — no pages yet ──────────────────────────────────────────
    # If there are no pages we skip training entirely and just save the raw
    # pretrained model. This is Round 0 — the model before it learned anything.
    # It gives us a honest baseline to compare all future rounds against.
    if not texts:
        print("\nNo pages found in data/pages/ — saving raw pretrained baseline.")
        print("This is Round 0 — the model knows English but nothing about your topic.\n")

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        run_name = f"round-{round_num}-baseline-0pages"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("round",       round_num)
            mlflow.log_param("pages",       0)
            mlflow.log_param("model",       MODEL_NAME)
            mlflow.log_param("baseline",    True)
            mlflow.log_metric("train_loss", 0)

        print(f"Baseline model saved to: {output_dir}")
        print(f"MLflow run: {run_name}")
        print(f"\nNext step: python src/serve.py")
        return

    # ── Normal training mode ───────────────────────────────────────────────────
    print(f"\nRound {round_num} — training on {len(texts)} page(s)")
    for f in files:
        print(f"  {os.path.basename(f)}")

    dataset = chunk_texts(texts, tokenizer)
    print(f"Created {len(dataset)} text chunks")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    run_name = f"round-{round_num}-{len(texts)}pages-{epochs}epochs"
    print(f"\nStarting run: {run_name}")
    print("Training on CPU — this will take a few minutes...\n")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("round",      round_num)
        mlflow.log_param("pages",      len(texts))
        mlflow.log_param("chunks",     len(dataset))
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model",      MODEL_NAME)
        mlflow.log_param("baseline",   False)

        result     = trainer.train()
        train_loss = result.training_loss
        mlflow.log_metric("train_loss", train_loss)
        mlflow.pytorch.log_model(model, name="student-model")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n── Round {round_num} complete ────────────────────────────")
        print(f"Pages        : {len(texts)}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Model saved  : {output_dir}")
        print(f"MLflow run   : {run_name}")
        print(f"\nNext step: python src/evaluate.py --model-dir {output_dir} --round {round_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int, default=0)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args.round, args.epochs, args.batch_size)