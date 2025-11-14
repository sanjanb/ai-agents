---
title: Prompt Engineering
summary: Techniques, patterns, and practical examples for eliciting reliable, useful outputs from LLMs.
---

## Chapter 2 — Prompt Engineering

This chapter covers practical prompt engineering: core techniques, advanced reasoning patterns, code-focused applications, and best practices you can apply when building agents.

### 1. Foundational Concepts

- Prompt engineering is an iterative design process. Small changes (wording, order, examples) can substantially change model outputs.
- Consider modality (text vs multimodal), token limits (cost/performance trade-offs), and sampling controls (temperature, top-k, top-p).

Recommended starting sampling settings for experiments:

- temperature: 0.0 — deterministic; good for logic, code, and precise formats.
- temperature: 0.5–0.8 — creative tasks (brainstorming, drafting).
- top_p: 0.9–0.95 and top_k: 0–50 as complementary controls.

Be aware of repetition loops at extreme settings — reduce temperature or use presence/penalty tokens to discourage loops.

### 2. Core Prompting Techniques

- Zero-shot: give the model a clear task description and input. Useful for quick one-off tasks.
- One-shot / Few-shot: include 1–5 high-quality examples demonstrating the desired input → output mapping.
- System prompts: provide persistent context ("You are a helpful code reviewer..."). Use system-level context to set role, tone, and constraints.
- Role prompting: attach personas (e.g., "You are a senior ML engineer") to change style and depth.
- Contextual prompting: include relevant documents, code snippets, or error traces to ground the response.
- Stepback prompting: start with a broad question to surface assumptions, then narrow for concrete steps.

Example: JSON schema enforcement (useful when downstream code must parse the response)

```text
System: You are a JSON-only assistant. Always respond with valid JSON matching the schema.
User: Convert the text below into the schema {"name": string, "tags": [string], "priority": number}.
Text: "Fix the login bug, tag: bug, auth; priority high"
```

### 3. Advanced Reasoning Techniques

- Chain-of-Thought (CoT): ask the model to show reasoning steps. Useful for complex decisioning and debugging. Costs increase with longer outputs.
- Self-Consistency: sample multiple CoT outputs and take a consensus answer to improve reliability.
- Tree of Thoughts (ToT): explore branching reasoning paths and prune based on heuristics — good for search/optimization.
- ReAct (Reason + Act): combine reasoning with tool calls (search, calculators, APIs) to produce grounded, up-to-date answers.
- Automatic Prompt Engineering (APE): programmatic generation & evaluation of many prompt variants to discover robust prompts automatically.

Quick CoT template:

"Begin by outlining the high-level approach in 2–4 bullets. Then write step-by-step reasoning. Finally, produce the final answer clearly labeled."

Mermaid (flow) — high level CoT → answer flow:

```mermaid
flowchart TD
  U[User Query] --> M[Model (CoT)]
  M -->|steps| A[Candidate Answers]
  A --> V[Validation / Self-Consistency]
  V --> Final[Final Answer]
```

### 4. Practical, Code-Focused Examples

OpenAI / HTTP style (pseudocode):

```python
# Minimal example structure (pseudocode)
prompt = "You are a helpful assistant. Return only valid JSON: {name, tags, priority}.\nText: '...'")
resp = client.generate(prompt, max_tokens=200, temperature=0.0)
data = json.loads(resp.text)
```

Few-shot for code translation (example):

Prompt includes 2–3 input/output pairs for converting Python functions → JavaScript. Make examples concise and cover edge cases.

Debugging pattern:

1. Provide the full traceback or failing snippet.
2. Ask for root cause, concise fix, and a one-line test to validate.

### 5. Best Practices and Patterns

- Keep prompts simple and specific. Prefer positive instructions (what to do) over negatives (what not to do).
- Use schemas (JSON, YAML) when you need machine-parseable responses.
- Iterate: save prompt versions, test with representative inputs, and measure outputs.
- Instrument prompts for evaluation: include unit tests, validation checks and fuzz inputs.
- Share and document prompts (a small prompt library) so teams can reuse effective patterns.

Checklist before moving to production:

- Does the prompt produce valid, parseable outputs for diverse inputs?
- Are sampling settings appropriate (deterministic for validation, creative for ideation)?
- Are safety/guardrails in place (content filters, rate limits, verification)?

### 6. Exercises

1. Zero-shot vs Few-shot: create a task (e.g., translate unit test names) and compare outputs for zero-shot and 3-example few-shot prompts. Measure accuracy.
2. CoT debugging: give a short erroneous function and ask the model to list steps it would take to debug, then produce the fix.
3. Schema enforcement: create a system prompt that **always** returns valid JSON for a given schema and test using varied inputs.

### References and Further Reading

- See `quick-knowledge/2. Prompt-engineering/` for the source notes and deeper exercises.

---

This page is intentionally practical — pick one pattern, try it, record the outcome, and iterate.
