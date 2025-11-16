import os
import argparse
import time
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def collate(batch, tokenizer):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
    return Batch(enc.input_ids, enc.attention_mask, labels)


def load_subset(sample_size: int):
    ds = load_dataset("newsgroup", split="train")  # lightweight community dataset
    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(sample_size))
    # map to expected fields
    ds = ds.rename_columns({"article": "text"})
    ds = ds.class_encode_column("topic")
    ds = ds.rename_column("topic", "label")
    return ds


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = load_subset(args.sample_size)
    num_labels = dataset.features["label"].num_classes

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.1, bias="none")
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    # split
    val_frac = 0.1
    val_count = int(len(dataset) * val_frac)
    val_ds = dataset.select(range(val_count))
    train_ds = dataset.select(range(val_count, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, tokenizer))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_val = 0.0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device),
                        labels=batch.labels.to(device))
            out.loss.backward()
            optimizer.step()
            scheduler.step()
        # validation
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                gold.extend(batch.labels.tolist())
        acc = accuracy_score(gold, preds)
        print(f"Epoch {epoch+1}: val accuracy={acc:.4f}")
        if acc > best_val:
            best_val = acc
            save_dir = os.path.join("adapters", "news_lora")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved adapter to {save_dir}")

    duration = time.time() - start_time
    print(f"Training complete. Best val acc={best_val:.4f}. Time={duration/60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune DistilBERT on newsgroups subset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample_size", type=int, default=1000, help="Subset size for quick runs")
    args = parser.parse_args()
    train(args)
