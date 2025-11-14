---
layout: default
title: 🚀 Quick Start With GitHub Pages
nav_order: 1
---

# 🚀 Quickstart — Preview docs & run examples

This quickstart helps you preview the documentation locally and run the minimal agent examples included in this repo. Use the MkDocs preview for fast editing and the small runnable examples to experiment with LLM-driven agents.

Prerequisites

- Python 3.8+ installed (this repository was validated with Python 3.13 on Windows).
- (Optional) git if you want to clone the repository.

1. Preview the docs locally (recommended)

Open PowerShell in the repository root and create a lightweight virtualenv (recommended):

```powershell
# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install mkdocs and the readthedocs theme (local preview)
python -m pip install --upgrade pip
python -m pip install mkdocs

# start the live preview (open http://127.0.0.1:8000)
python -m mkdocs serve -a 127.0.0.1:8000
```

Notes:

- If `Activate.ps1` is blocked by your PowerShell execution policy, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once (requires administrator permission depending on policy).
- The docs are in the `docs/` folder. Editing files there will hot-reload the preview.

2. Run the minimal agent example (a tiny, local policy)

Create a file named `agent.py` with this minimal example (or copy from `docs/agents/getting-started.md`):

```python
def simple_agent(observation: str) -> str:
    if 'hello' in observation.lower():
        return 'Hi — how can I help?'
    if 'compute' in observation.lower():
        return 'I can compute simple math: e.g. 2+2=4'
    return 'I did not understand your request.'

if __name__ == '__main__':
    obs = input('Observation> ')
    print('Action:', simple_agent(obs))
```

Then run:

```powershell
python agent.py
```

3. Try a small LLM-backed example (Hugging Face)

If you want to experiment with model generation, install `transformers` and a small model:

```powershell
python -m pip install transformers torch

# run a tiny demo (replace in a script or REPL)
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
prompt = 'Write a concise definition of an AI agent.'
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
outputs = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.9, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
PY
```

Notes on credentials & APIs

- If you use a hosted LLM provider, store API keys in environment variables and follow the provider's SDK guidance. Never commit secrets into the repo.

Where to go next

- AI Agents → Getting Started — run the first minimal agent and extend it.
- AI Agents → Foundational LLMs — deep dive into tokens, architecture, training and decoding.
- Examples: I can scaffold an `examples/` folder with runnable demos (Hugging Face, RAG, or small RLHF toy pipelines) — tell me which and I'll add it.

Troubleshooting

- If MkDocs reports YAML/indentation errors, check `mkdocs.yml` for tabs vs spaces (YAML requires spaces).
- If model downloads fail, ensure you have network access and enough disk space for the model.

---

If you want, I can now scaffold `examples/agents/` with the Hugging Face demo and a `requirements.txt`. Say "scaffold examples" and I'll add them and validate the build.
