---
title: Foundational LLMs & Text Generation
---

# Foundational LLMs & Text Generation

This chapter explains the foundations of large language models (LLMs) and text generation. It covers the core concepts, model architectures, training objectives, decoding strategies, safety considerations, and practical examples you can run locally or with hosted APIs.

The structure below mirrors the provided whitepaper while adding diagrams, runnable snippets, and clarifying explanations to help readers build intuition and practical skills.

## Chapter structure (what you'll learn)

- Motivation: why LLMs are central to modern AI systems
- Key concepts: tokens, tokenization, context windows, prompts
- Architecture: the transformer building blocks and how they enable language modeling
- Training objectives: causal LM, masked LM, denoising and instruction tuning
- Decoding & generation: greedy, beam, sampling, temperature, nucleus/top-p
- Fine-tuning & alignment: supervised fine-tuning, instruction tuning, RLHF
- Evaluation & metrics: perplexity, human evaluation, automated metrics
- Safety, robustness and limitations
- Practical examples: Hugging Face and LLM API patterns
- Exercises and further reading

---

## 1. Motivation

Large language models learn statistical patterns of language from huge corpora. They provide a flexible, general-purpose capability: given a sequence of text (the prompt), predict the next tokens. This simple capability scales into powerful behaviors: summarization, translation, question-answering, code completion, and more. In agent systems, LLMs often serve as the "policy" or reasoning component that maps observations and memory to actions.

Practical goals for this chapter:

- Build intuition about why transformers work for language tasks.
- Understand trade-offs in model size, latency, and cost.
- Learn safe and reproducible ways to use LLMs in experiments.

## 2. Key concepts and terminology

### Tokens and tokenization

- A token is the atomic unit the model predicts (subword, byte pair, or byte-level unit). Tokenization converts text into tokens.
- Common tokenizers: Byte-Pair Encoding (BPE), WordPiece, byte-level BPE. Each trades off vocabulary size and slice granularity.
- Token count matters: context length and billing (API) are measured in tokens.

Visual: tokenization example

```text
Text:  "Large language models are powerful."
Tokens: ["Large", " language", " models", " are", " powerful", "."]  (example using subword tokenizer)
```

### Context window & prompt

- Context window: maximum number of tokens a model can see at once (e.g., 2k, 8k, 32k+). Longer windows allow multi-document reasoning but increase compute and memory.
- Prompt engineering: designing the input (instructions, examples, system messages) to elicit desired behavior — includes few-shot prompting (examples in prompt) and zero-shot prompting.

### Pretraining vs fine-tuning

- Pretraining: train a model on large corpora with a self-supervised objective (predict next token or masked tokens).
- Fine-tuning: adapt the pretrained model to a task using labeled examples or specialized objectives.

### Perplexity and other evaluation concepts

- Perplexity: a measure of how well a probabilistic model predicts a sample. Lower is better; not always correlated with downstream task performance.
- Other metrics: BLEU/ROUGE (n-gram overlap), BERTScore, and human evaluation (often the gold standard).

## 3. Transformer architecture (high level)

Transformers are the dominant architecture for LLMs. Key pieces:

- Token embeddings + positional encodings
- Self-attention: pairwise token interactions to compute context-aware representations
- Feed-forward blocks: local transformations per position
- Stacking layers and residual connections

Mermaid diagram — simplified transformer encoder/decoder flow:

```mermaid
flowchart LR
    A[Input tokens] --> B[Embedding + Positional Encoding]
    B --> C(Self-attention)
    C --> D(Feed-forward)
    D --> E[Output logits]
```

Notes:

- For causal language modeling (auto-regressive generation) we use a decoder-only transformer (masked self-attention to prevent seeing future tokens).
- For bidirectional tasks (masked LM), encoder or encoder-decoder variants are used.

## 4. Training objectives

### Causal LM (auto-regressive)

- Objective: maximize likelihood of next token given previous tokens. Used by GPT-style models.

Mathematically:

$$\\mathcal{L} = -\\sum_{t=1}^T \\log p(x_t | x_{<t})$$

### Masked LM / Denoising

- Mask tokens and predict them (BERT), or use corrupted inputs and learn to reconstruct (BART, T5). These support bidirectional context.

### Instruction tuning & supervised fine-tuning

- Supervised finetuning uses labeled input-output pairs (prompt → desired output) to align models with tasks.

### Reinforcement Learning from Human Feedback (RLHF)

- RLHF pipeline (simplified):
    1. Collect model outputs for prompts.
 2. Humans rank outputs.
 3. Train a reward model from rankings.
 4. Use RL (PPO) to optimize policy against the reward model.

Mermaid sequence diagram (RLHF simplified):

```mermaid
sequenceDiagram
    participant Model
    participant Human
    participant RewardModel
    participant RL

    Model->>Human: generate responses
    Human->>RewardModel: ranking data
    RewardModel->>RL: provide rewards
    RL->>Model: update policy
```

## 5. Decoding & generation strategies

After the model produces logits for the next token, decoding turns those logits into tokens. Choice of decoding greatly affects output quality.

- Greedy: pick highest-probability token. Fast, often repetitive.
- Beam search: keep top-k hypotheses, expands search; good for deterministic tasks but can be costly and produce generic outputs.
- Sampling: sample from the distribution to increase diversity.
    - Temperature: scale logits to make distribution sharper (low T) or flatter (high T).
    - Top-k: sample from top k tokens.
    - Top-p (nucleus): sample from smallest set whose cumulative probability >= p.

Best practice: use sampling with tuned temperature and top-p for creative generation; use beam or greedy for structured predictions.

## 6. Common models & toolkits

- GPT-family: decoder-only causal LMs (GPT-2, GPT-3, GPT-4 — closed or partially closed variants)
- LLaMA, OPT, BLOOM: open-source large models with various licenses
- PaLM, Chinchilla: research-scale models with variants
- Toolkits: Hugging Face Transformers, Hugging Face Accelerate, transformers + PEFT (parameter-efficient fine-tuning), OpenAI SDKs, LangChain for orchestration

## 7. Practical examples (runnable)

Below are two minimal examples: (A) using Hugging Face local model (small) and (B) calling an LLM API. These are intentionally small so they run quickly.

### A — Minimal local generation with Hugging Face (transformers)

Requirements: `pip install transformers torch --upgrade` (or use CPU-only torch)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2'  # small model for demos
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Write a concise definition of an AI agent."
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

# generate (sampling)
outputs = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.9, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Notes:

- For larger models, follow Hugging Face guides for `accelerate` and model sharding. Replace `gpt2` with a local checkpoint as needed.

### B — Minimal API-style pseudocode (replace with your provider's SDK)

```python
import requests

API_KEY = 'REPLACE_ME'
API_URL = 'https://api.example.com/v1/generate'

def call_llm_api(prompt: str):
        payload = { 'prompt': prompt, 'max_tokens': 150 }
        headers = {'Authorization': f'Bearer {API_KEY}'}
        resp = requests.post(API_URL, json=payload, headers=headers)
        return resp.json()['text']

print(call_llm_api('Summarize the design principles for AI agents in one paragraph.'))
```

Replace with the specific SDK (OpenAI, Anthropic, Google, etc.) and follow provider rate-limits and credential handling guidelines.

## 8. Fine-tuning and efficient adaptation

- Full fine-tuning: update all parameters — effective but expensive for large models.
- Parameter-efficient tuning: LoRA, adapters, and prefix tuning change fewer parameters and are cheaper for multiple tasks.
- Distillation: train smaller student models to mimic larger teacher models for faster inference.

## 9. Evaluation

- Automatic metrics: perplexity (language modeling), BLEU/ROUGE (overlap), BERTScore (semantic similarity).
- Human evaluation: pairwise preference, Likert-scale ratings, task-specific success metrics.
- Robustness tests: adversarial prompts, stress tests across edge cases.

## 10. Safety, biases & limitations

- Hallucinations: models can invent false facts — mitigation includes retrieval augmentation, verification steps, and uncertainty prompts.
- Bias and fairness: models can reflect biases in training data — perform dataset audits, controlled generation, and use fairness evaluations.
- Cost and latency: large models are expensive; choose model size for the use-case, and apply quantization/compilation for faster inference.

Mitigations and best practices

- Use retrieval-augmented generation (RAG) when factual accuracy is required.
- Validate outputs with rule-based checks or secondary models.
- Rate-limit and sanitize user inputs.

## 11. Visual summary (high-level pipeline)

```mermaid
flowchart TD
    A[User / Agent] --> B[Prompt / Observation]
    B --> C[LLM (pretrained + adapted)]
    C --> D[Decoding (sampling/beam)]
    D --> E[Action / Generated text]
    E --> F[Post-processing, verification, safety checks]
    F --> G[Agent executes or replies]
```

## 12. Exercises

1. Run the Hugging Face example, vary `top_p` and `temperature`, and compare outputs.
2. Replace the local model with a small on-disk instruction-tuned model and measure quality differences.
3. Implement a simple RAG loop: retrieve a short document and prepend it to the prompt before generation; compare factual accuracy.

## 13. Further reading and references

- Vaswani et al., "Attention Is All You Need" (transformers)
- Radford et al., GPT papers
- Brown et al., GPT-3
- Raffel et al., T5; Lewis et al., BART
- Papers and resources referenced in the supplied whitepaper (see `assets/whitepapers/`)

---

Author notes

- If you paste sections from the whitepaper you'd like included verbatim, I can insert them under the appropriate headings and add figure references.
- I can also generate an example `examples/agents/` folder with runnable scripts and a `requirements.txt` if you want reproducible demos.

