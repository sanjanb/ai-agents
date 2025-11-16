import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(sample_size):
    ds = load_dataset("newsgroup", split="test")
    ds = ds.rename_columns({"article": "text"})
    ds = ds.class_encode_column("topic")
    ds = ds.rename_column("topic", "label")
    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=123).select(range(sample_size))
    return ds


def evaluate(model, tokenizer, dataset, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for example in dataset:
            enc = tokenizer(example["text"], truncation=True, max_length=256, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            pred = torch.argmax(logits, dim=-1).item()
            preds.append(pred)
            labels.append(example["label"])
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return acc, cm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = load_data(args.sample_size)

    # Baseline
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=20).to(device)
    base_acc, base_cm = evaluate(base_model, tokenizer, dataset, device)
    print(f"Baseline accuracy: {base_acc:.4f}")

    # Adapter
    adapter_dir = args.adapter_dir
    if adapter_dir:
        adapted = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=20)
        adapted = PeftModel.from_pretrained(adapted, adapter_dir)
        adapted.to(device)
        ada_acc, ada_cm = evaluate(adapted, tokenizer, dataset, device)
        print(f"Adapter accuracy:  {ada_acc:.4f}")
        improvement = ada_acc - base_acc
        print(f"Improvement:      {improvement:.4f}")
        print("Confusion Matrix (Adapter):")
        print(ada_cm)
    else:
        print("No adapter_dir provided; skipping adapted evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline vs LoRA adapter on newsgroups classification")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--adapter_dir", type=str, default="adapters/news_lora")
    args = parser.parse_args()
    main(args)
