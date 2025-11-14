---
layout: default
title: Local Development Index Page
nav_order: 1
---

---

layout: default
title: AI Agents — Book Index (Template)
nav_order: 1

---

# AI Agents — Documentation (Book Index)

Welcome to the AI Agents documentation. This page is a book-style index template used when previewing the Jekyll site from the repository root. It is intended as a high-level table of contents and reading guide for the materials in `docs/`.

Use this template as a starting point — once the chapter content is final we will update the links and the order to match the finalized curriculum.

## Book metadata

- Title: AI Agents — Understanding & Building Practical Agents
- Editor: (your name here)
- Version: draft
- Status: in-progress

## Quick start (recommended reading order)

1. Introduction & Overview — start here to understand goals and scope.
2. Foundational LLMs & Text Generation — deep dive into chapter 1.
3. Getting Started — run the minimal agent examples and try the hands-on exercises.
4. Examples & Patterns — RAG, instruction tuning, PEFT, and small RLHF pipelines.
5. Evaluation & Safety — tests, metrics, and deployment hygiene.

## Table of contents (links)

- [AI Agents — Overview](/docs/agents/)
- [Foundational LLMs & Text Generation](/docs/agents/foundational-llms/)
- [Getting Started — Minimal Agent](/docs/agents/getting-started/)
- [Quickstart](/docs/quickstart/)
- [Configuration reference](/docs/configuration/configyml/)
- [Demo pages & examples](/docs/demo-pages/)

> Note: when previewing with MkDocs the site root is served from `docs/` and links above work as relative paths; when serving with Jekyll from the repo root the same links resolve to the generated site paths.

## How to preview this book locally

- Quick MkDocs preview (recommended for authoring):

```powershell
# from repository root
python -m mkdocs serve -a 127.0.0.1:8000
# open http://127.0.0.1:8000
```

- Jekyll preview (to see the theme/layout from root):

```powershell
bundle install
bundle exec jekyll serve --host 0.0.0.0 --watch
# open http://127.0.0.1:4000
```

## Authors' checklist (for each chapter)

- Title, abstract and learning objectives
- A short motivating example or use-case
- Key concepts and diagrams (add Mermaid where helpful)
- Runnable example(s) and a `requirements.txt` where applicable
- Exercises and answers (or example solutions)
- References and further reading

## Contribution & publishing notes

- Keep text clear and modular — chapters should be independently runnable where possible.
- For code examples, prefer small reproducible snippets and link to `examples/` for full scripts.
- Before merging a chapter, run `mkdocs build` and ensure no warnings for YAML/links.

---

This is a template index — feel free to edit the metadata, add a cover image, or adjust the reading order. When you want, I can replace the placeholders with final content and add a cover/hero SVG.
