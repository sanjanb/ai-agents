# Examples: Generative Agents

This folder contains runnable examples referenced by Chapter 3: Generative Agents.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r examples\agents\requirements.txt
```

## 1) Gemini Function Calling (SQLite)

- File: `examples/agents/gemini_function_calling.py`
- Requirements: `GOOGLE_API_KEY` environment variable.

```powershell
$env:GOOGLE_API_KEY = "<your_api_key>"
python examples\agents\gemini_function_calling.py
```

## 2) LangGraph ReAct Agent (with offline fallback)

- File: `examples/agents/langgraph_react_agent.py`
- If `OPENAI_API_KEY` is set, uses a real model via LangChain OpenAI.
- Otherwise, falls back to a mock policy so it runs offline.

```powershell
# Optional
$env:OPENAI_API_KEY = "<your_openai_key>"
python examples\agents\langgraph_react_agent.py
```

## 3) RAG Memory Agent (FAISS)

- File: `examples/agents/rag_memory_agent.py`
- Builds a small FAISS index and retrieves semantically similar chunks.

```powershell
python examples\agents\rag_memory_agent.py
```

## 4) Evaluation Harness for RAG

- File: `scripts/eval_rag.py`
- Computes Precision@k and MRR. If no dataset is provided, uses a tiny built-in set.

```powershell
python scripts\eval_rag.py --k 5
```

Troubleshooting:

- If model downloads fail for `sentence-transformers`, ensure internet access or pre-download models.
- For `faiss-cpu` on Windows, use a Python 3.10–3.12 environment for best compatibility.
- If you hit API limits, reduce batch sizes and add retries.
