---
title: Getting started with AI agents
---

# Getting started — Minimal agent

This guide walks through a minimal, educational AI agent implemented in Python. The goal is to show the structure and how to run it locally.

Agent contract (tiny):

- Input: environment observation (string)
- Output: action (string)

Example agent (pseudo/real):

```python
def simple_agent(observation: str) -> str:
    """A toy agent that echoes requests or responds to a simple instruction."""
    if 'hello' in observation.lower():
        return 'Hi — how can I help?'
    if 'compute' in observation.lower():
        return 'I can compute simple math: e.g. 2+2=4'
    return 'I did not understand your request.'


if __name__ == '__main__':
    obs = input('Observation> ')
    print('Action:', simple_agent(obs))
```

How to run locally

1. Save the snippet above to a file named `agent.py`.
2. Run with Python 3.8+:

```bash
python agent.py
```

What to try next

- Replace the policy with a model inference (e.g., a small transformer or an LLM API call).
- Add an environment loop that provides observations and records actions.
- Add tests for expected behaviours (hello -> greeting).

Notes on safety

- Keep prompts and model queries small during experiments.
- Sanitize and validate inputs when connecting to external APIs.
