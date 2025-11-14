---
layout: default
title: Index
nav_exclude: true
---

# AI Agents — Learn to understand and build intelligent systems

Welcome. This site documents practical, reproducible approaches to designing, building and evaluating AI agents — systems that sense, reason, and act using machine learning components (especially LLMs).

What you'll find here

- A guided learning path for AI agents: concepts, architectures, and worked examples.
- Hands-on tutorials and runnable code (Hugging Face, local models, and API patterns).
- A focused first chapter on foundational LLMs & text generation (see "AI Agents → Foundational LLMs").

Quick links

- AI Agents overview — Agents landing page: /agents/
- Getting started — Minimal runnable agent: /agents/getting-started/
- Foundational LLMs — Chapter 1: /agents/foundational-llms/

How to preview locally

1. With MkDocs (recommended for docs preview):

```powershell
# from repository root
python -m mkdocs serve -a 127.0.0.1:8000
# open http://127.0.0.1:8000 in a browser
```

2. With Jekyll (if you want to preview the original theme layout):

```powershell
bundle install
bundle exec jekyll serve --host 0.0.0.0 --watch
```

Contributing

- Found an error or want to add an example? Open a PR with a short description and runnable code where applicable.
- If you want CI previews, I can add a GitHub Actions workflow to build MkDocs on PRs and post a preview link.

Support & notes

- This site mixes MkDocs previews and Jekyll theme sources for documentation development. The canonical learning content is in `docs/` (used by MkDocs).
- If you prefer a different learning path or additional example languages (JS/TS, Rust), tell me and I'll add them.

---

Start with "AI Agents → Getting Started" for a minimal agent you can run locally, or open "AI Agents → Foundational LLMs" to dive into chapter 1.
