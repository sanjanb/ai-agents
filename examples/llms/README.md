# Domain-Specific LLM Examples

This folder provides runnable examples supporting Chapter 4 (Domain-Specific LLMs):

## Contents

- `requirements.txt` – Dependencies (transformers, peft, datasets, bitsandbytes (optional GPU), google-genai for grounding demos).
- `lora_finetune_news.py` – LoRA PEFT fine-tuning of DistilBERT on a subset of newsgroups for fast experimentation.
- `eval_domain_classification.py` – Compares baseline model performance vs LoRA adapter.
- `search_grounding_stub.py` – Sketch of search grounding pipeline (query planning, retrieval, synthesis).

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\activate
pip install -r examples/llms/requirements.txt
```

If you lack a GPU or bitsandbytes fails to build, you can remove `bitsandbytes` from the requirements and still run CPU experiments (slower).

## LoRA Fine-Tuning

```powershell
python examples/llms/lora_finetune_news.py --epochs 1 --sample_size 1000
```

Artifacts saved under `adapters/news_lora`.

## Evaluate Adapter

```powershell
python examples/llms/eval_domain_classification.py --sample_size 500 --adapter_dir adapters/news_lora
```

## Grounding Demo (Stub)

Replace placeholder `web_search` with a real search API (e.g., Google Custom Search). Ensure environment variables for keys are set before calling.

```powershell
python examples/llms/search_grounding_stub.py
```

## Next Steps

- Add reranking (cross-encoder) for evidence ordering.
- Implement real search grounding with API responses.
- Introduce self-reflection loop to verify citations density.
- Containerize fine-tuning workflow for reproducibility.
