# Domain-Specific LLMs

> Chapter 4 – Adapting, Grounding & Operating Large Language Models for Specific Domains

## 1. Motivation & Landscape

General-purpose frontier models are powerful but often overkill (and expensive) for narrow domains: compliance analytics, biomedical abstracts, legal clause extraction, support ticket routing, fintech risk narratives. Domain adaptation aims for: higher accuracy, lower latency/cost, better controllability, improved factuality, and reduced hallucination via grounding.

Key Drivers:

- Data Specialization: Exploit in-domain terminology, style, decision heuristics.
- Governance: Constrain responses (policy, PII, safety) more easily with predictable behavior.
- Efficiency: Smaller fine‑tuned or instruction-adapted checkpoints vs calling large general APIs.
- Reliability: Integrate verifiable sources (search, internal docs) to increase trust.

## 2. Adaptation Strategy Taxonomy

| Strategy                              | When to Use                                               | Pros                              | Cons                                         |
| ------------------------------------- | --------------------------------------------------------- | --------------------------------- | -------------------------------------------- |
| Prompt Engineering / System Design    | Early exploration, no training budget                     | Fast iteration                    | Ceiling on quality                           |
| Retrieval Augmented Generation (RAG)  | Frequent need for fresh / large corpora facts             | Up-to-date, auditable sources     | Requires solid indexing & evaluation         |
| Structured Tool / API Grounding       | Tasks needing external authoritative data (search, DB)    | Verifiability, traceability       | Tool latency & failure modes                 |
| Lightweight Fine-Tuning (LoRA / PEFT) | Need style/format consistency or specialty classification | Lower cost vs full FT             | Possible overfitting if low data             |
| Full Fine-Tuning                      | Large domain dataset; major distribution shift            | Highest potential task fit        | Expensive; risk catastrophic forgetting      |
| Synthetic Data Augmentation           | Scarce labeled data; need coverage                        | Scales examples, edge cases       | Quality filtering required                   |
| Multi-Agent Orchestration             | Complex workflows (plan, reflect, verify)                 | Process yields better reliability | Higher complexity, requires evaluation loops |

Decision Heuristics:

1. Start with RAG + prompt design. Measure gaps (accuracy, consistency, latency).
2. If style/format or classification errors persist → try LoRA.
3. If large domain dataset & significant semantic shift → consider full FT.
4. Always keep grounding (search / internal doc retrieval) for factual tasks.

## 3. Data Quality & Curation

Dimensions:

- Coverage: Representative domain subtopics, edge cases, adversarial formats.
- Signal Clarity: Minimize noisy labels; prefer canonical references.
- Structure: Use JSON/TSV with explicit fields for classification / extraction.
- Versioning: Tag dataset snapshots (e.g., `finance-risks-v2024Q4`).
- Governance: Redact PII; maintain provenance metadata.

Practical Checks:

- Duplicate ratio < 5%.
- Class imbalance ratio < 10x between majority/minority if possible (else use weighting).
- Split by time (avoid leakage) for evolving domains.

Synthetic Data Use:

- Generate candidate examples with base model; filter via: perplexity range, rule validators, classifier confidence, dedup similarity (<0.9 cosine).

## 4. Grounding & External Knowledge Integration

Grounding = model augments its reasoning with authoritative sources (search engine, internal KB, database). Workflow:

1. Query Planning (LLM decides search queries).
2. Retrieval (Search API / Vector store / DB).
3. Evidence Selection (ranking, dedup, trimming tokens).
4. Answer Synthesis with citations.
5. Optional Verification (self-reflection or secondary model).

Benefits:

- Factuality: Source-linked outputs.
- Transparency: Provide snippet + URL / doc ID.
- Robustness: Easier to update knowledge base than re-train.

Implementation Modalities:

- Web Search Grounding (e.g., Google Search). Use rate limiting & query caching.
- Internal Corpus RAG: FAISS / dedicated vector DB with chunk embeddings, metadata filters.
- Hybrid: First web grounding for fresh facts, fallback to internal RAG for proprietary guidance.

Quality Levers:

- Chunk Size & Overlap (semantic coherence vs recall).
- Reranking (cross-encoder or embedding similarity + diversity penalty).
- Citation Formatting (structured JSON for downstream auditing).

## 5. Fine-Tuning & PEFT (LoRA)

Lightweight adapters (LoRA) inject low-rank matrices into attention weights enabling rapid adaptation:

- Keeps frozen base weights → reduced risk of catastrophic forgetting.
- Few trainable params (<<1%) → cheaper, faster.
- Combine multiple domain adapters (multi-LoRA) and select at inference.

Typical Workflow:

1. Select Base Model (size vs latency trade-off).
2. Prepare Dataset (JSONL; fields: `input`, `output`, optional `metadata`).
3. Tokenize & Split (train/val/test).
4. Configure LoRA (r, alpha, dropout). Start small (r=8–16).
5. Train with early stopping on validation loss or task metric.
6. Evaluate: exact match / F1 / accuracy / calibration.
7. Package adapter & store with model card + data lineage.

Full Fine-Tuning Considerations:

- Mixed precision + gradient checkpointing for memory.
- Regularization: weight decay, dropout, low learning rate.
- Mitigate forgetting: interleave small % of original (general) data.

## 6. Evaluation Framework

Multi-dimensional evaluation beyond a single metric:

- Task Performance: Accuracy, F1, ROUGE, BLEU depending on task.
- Retrieval Quality (for RAG): Precision@k, MRR (see existing `scripts/eval_rag.py`).
- Factuality: Automated citation presence + human spot checks.
- Robustness: Perturb inputs (typos, synonym swaps) measure drop.
- Drift: Track monthly accuracy vs dataset changes.
- Cost: Tokens / second; adapter training cost vs improvement.

Set a Promotion Gate:

```
if (task_accuracy_gain >= 5% AND hallucination_rate <= baseline - 10%) promote_adapter
else iterate (data cleaning / more grounding / refine LoRA params)
```

## 7. Operational Patterns & Monitoring

- Versioning: `model-base@1.0`, `adapter-claims@2024-11-15`.
- Rollback: Keep previous adapter; implement A/B switch via config flag.
- Telemetry: Log retrieval latency, empty-result rate, citation count, user feedback labels.
- Guardrails: Regex / semantic filters before final output (PII, toxicity). If blocked → regeneration w/ constraints.
- Drift Alerts: Control charts on key metrics (e.g., accuracy weekly). Threshold triggers re-evaluation or partial re-training.

## 8. Putting It Together (Workflow Blueprint)

```
User Query
  → Classify Intent (domain or general?)
     → If domain → Retrieve (vector + optional web grounding)
        → Build Evidence Set (rerank, diversify)
           → Compose Prompt (system + task + evidence JSON)
              → Inference (base + domain adapter)
                 → Verification (self-check + rule filters)
                    → Respond + Citations + Structured Metadata
```

## 9. Case Study (Illustrative)

Domain: Financial Risk Summaries.

- Start: General model hallucinating outdated regulation numbers.
- Add RAG: Index latest regulatory circulars → reduces incorrect citations by 40%.
- LoRA Adapter: Fine-tune on 3k curated analyst summaries → style consistency +15% readability score.
- Monitoring: Weekly drift detected as new regs published; re-index only (no retrain) → maintain factuality.
- Outcome: Cost per answer down 30% (smaller base + adapter) with higher trust.

## 10. Hands-On Examples

Provided scripts under `examples/llms/`:

- `lora_finetune_news.py`: LoRA classification fine-tune on 20 Newsgroups subset.
- `eval_domain_classification.py`: Baseline vs adapter evaluation (accuracy, confusion matrix).
- `search_grounding_stub.py`: Demonstrates query planning + grounding call structure.

### Quick Start

```powershell
# Create environment
python -m venv .venv; .\.venv\Scripts\activate
pip install -r examples/llms/requirements.txt

# Run LoRA fine-tuning (CPU/GPU autodetect)
python examples/llms/lora_finetune_news.py --epochs 1 --sample_size 1000

# Evaluate baseline vs adapted
python examples/llms/eval_domain_classification.py --sample_size 500

# Inspect adapter directory
Get-ChildItem adapters\news_lora
```

### Grounding Pattern (Pseudo)

```python
queries = plan_queries(user_question)          # LLM planning
results = web_search(queries)                  # API results
ranked = rerank(results)                       # Evidence selection
answer = synthesize(user_question, ranked)     # Generate with citations
```

## 11. Extension Exercises

1. Add a reranking cross-encoder to improve retrieval quality.
2. Implement adversarial robustness test (typo noise injection) and measure drop.
3. Add a second LoRA adapter (legal) and dynamically select by intent classifier.
4. Integrate OpenTelemetry spans around retrieval & synthesis stages.
5. Add self-reflection loop: Have model critique its answer and regenerate if low citation density.

## 12. Checklist

- [ ] Data cleaned & versioned
- [ ] Retrieval index built & evaluated
- [ ] Adapter trained with validation improvement
- [ ] Grounding sources cited
- [ ] Monitoring metrics defined
- [ ] Rollback strategy documented

## 13. References & Further Reading

- LoRA paper (Hu et al. 2021)
- Retrieval-Augmented Generation (Lewis et al. 2020)
- Toolformer & function calling design paradigms
- PEFT libraries & Hugging Face ecosystem docs
- Google AI Studio grounding resources

---

_This chapter synthesizes internal conceptual resources on agentic workflows, grounding mechanics, evaluation harness patterns, and fine-tuning notebooks to present a structured approach to domain-specific LLM adaptation._
