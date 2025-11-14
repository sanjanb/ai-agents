---
layout: default
title: Install Theme In A Jekyll Project
nav_order: 2
---

# Install Theme In A Jekyll Project

**WARNING!** This theme hasn't been published as a Gem yet, so while these
instructions should be correct once it has been published, currently it won't
work.

---

You can add this port of the Read The Docs theme to your Jekyll project as
you would normally do with any other
[gem-based Jekyll theme](https://jekyllrb.com/docs/themes/).

There are two simple methods to do this:

1. Editing your project Gemfile
2. Manually installing the gem

## Edit your project Gemfile

Add this line to your Jekyll site's `Gemfile`:

```ruby
gem "jekyll-theme-rtd"
```

Add this line to your Jekyll site's `_config.yml`:

```yaml
theme: jekyll-theme-rtd
```

And then execute:

```bash
$ bundle
```

## Manually install gem

Or install the gem yourself with:

```bash
$ gem install jekyll-theme-rtd
```

And add this line to your Jekyll site's `_config.yml`:

```yaml
theme: jekyll-theme-rtd
```

## Local Jekyll to test GH Pages

If you are hosting your website in GH Pages, but testing locally with Jekyll
you might want to have a look at the official documentation for
[Testing your GitHub Pages site locally with Jekyll](https://help.github.com/en/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll).

---

## Book-style index template (use this page to author an index)

If you want a book-style index (a high-level table of contents and reading guide) that can be served from the repository root with Jekyll, copy the template below into `index.md` at the repository root. This is a lightweight template tailored to the AI Agents documentation; edit metadata and links to match your final chapter names.

```markdown
---
layout: default
title: AI Agents — Book Index
nav_order: 1
---

# AI Agents — Documentation (Book Index)

Welcome to the AI Agents documentation. This book-style index links to the main chapters, provides a recommended reading order, and lists authoring notes.

## Reading order (recommended)

1. Foundational LLMs & Text Generation
2. Getting Started — Minimal agent
3. Examples & Patterns (RAG, PEFT, small RLHF)
4. Evaluation & Safety

## Table of contents

- [Foundational LLMs & Text Generation](/docs/agents/foundational-llms/)
- [Getting Started — Minimal Agent](/docs/agents/getting-started/)
- [Quickstart](/docs/quickstart/)
- [Configuration reference](/docs/configuration/configyml/)

## Authors' checklist

- Title, abstract and learning objectives
- Small runnable example and `requirements.txt`
- Diagrams (Mermaid), exercises, and references

---

Edit and extend this template as needed. When you're ready, copy it to the repository root `index.md` so Jekyll will show the book index at the site root.
```
