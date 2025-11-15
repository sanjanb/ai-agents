# Generative Agents

This chapter explains what generative AI agents are, how they are architected, how they reason and use tools, and how to build them end-to-end with function calling and agent frameworks. It distills and integrates material from:

- quick-knowledge/3. Generative-agents/agents/gnerative-ai-agents.md
- quick-knowledge/3. Generative-agents/agents/concept-levels.md
- quick-knowledge/3. Generative-agents/agents/guide-through.md
- quick-knowledge/3. Generative-agents/agents/3.1-function-calling-with-the-gemini-api.ipynb
- quick-knowledge/3. Generative-agents/agents/3.2-building-an-agent-with-langgraph.ipynb
- quick-knowledge/3. Generative-agents/agents/22365_19_Agents_v8.pdf
- quick-knowledge/3. Generative-agents/agents-companion/Agents_Companion_v2 (3).pdf

## 1. What is a generative agent?

A generative agent is a goal-directed program that observes, reasons, and acts using a foundation model (LLM) plus external tools and memory. Unlike standalone LLMs (which only predict the next token), agents:

- Access live information (APIs, databases, search)
- Take actions (send emails, update records, trigger workflows)
- Maintain multi-turn state and memory
- Plan and self-reflect to reach goals

Use-cases include: research assistants, customer support copilots, growth/ops automations, data analysts, and autonomous testers.

## 2. Agent cognitive architecture

An effective mental model (from the provided resources):

- Model (the thinker): one or more LLMs (text-only or multimodal) that understand, plan, and generate.
- Tools (the doers): extensions, functions, and data stores that provide perception and action.
- Orchestration (the strategist): the controller that loops over reason → act → observe, manages memory, and maintains goal alignment.

Interface-style view:

- Inputs: user goals, constraints, environment signals
- Core loop: plan → call tools → observe → update plan → continue/stop
- Outputs: actions taken, results, and explanations

## 3. Reasoning frameworks

Reasoning becomes explicit via prompting and control flow:

- ReAct (Reason + Act): interleave thought, tool calls, and observations.
- Chain-of-Thought (CoT): produce step-by-step intermediate reasoning.
- Self-Consistency: sample multiple CoTs and vote.
- Tree-of-Thoughts (ToT): branch, explore, and select the best path.
- Planner-Executor: separate high-level planning from low-level acting.

Tip: Combine ReAct + CoT for transparency and debuggability; reserve ToT for hard, branching problems given its cost.

## 4. Tools: extensions, functions, and data stores

- Extensions (connectors): external APIs and services (e.g., search, flights, calendar). Standardize schemas and auth.
- Functions (local logic): safe, typed functions executed under your control (validation, sandboxing).
- Data Stores: RAG with embeddings + vector DBs to ground the model in your latest knowledge and structured filters.

Security notes:

- Validate and sanitize inputs/outputs
- Apply least-privilege API scopes and audit logs
- Keep a human-in-the-loop for sensitive actions

## 5. Memory, planning, and feedback

- Short-term state: the working context of the current task/loop.
- Long-term memory: facts, preferences, and prior outcomes indexed via embeddings for retrieval.
- Planning: hierarchical plans (goals → tasks → steps) with checkpoints.
- Reflection: self-critique and repair when confidence is low or errors occur.

Practical heuristics:

- Separate scratchpad (reasoning) from user-visible output
- Log tool calls and observations for traceability
- Use confidence thresholds to trigger more retrieval or human review

## 6. Function calling with the Gemini API (hands-on)

Below is a minimal, self-contained example of typed function calling that lets an agent query a local SQLite database. It mirrors the notebook in `3.1-function-calling-with-the-gemini-api.ipynb` while being framework-agnostic.

```python
# pip install google-genai==1.7.0 sqlite-utils
from typing import List, Dict, Any
from google import genai
from google.genai import types
import sqlite3

# 1) Define functions with clear types and docstrings
def list_tables() -> List[str]:
    """List available SQLite tables."""
    cur = db.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def describe_table(table: str) -> List[Dict[str, Any]]:
    """Describe columns for a given table."""
    cur = db.cursor()
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3]} for r in rows]

def run_sql(query: str) -> List[Dict[str, Any]]:
    """Execute a read-only SQL query and return rows as dicts."""
    assert query.strip().lower().startswith("select"), "Only SELECT allowed"
    cur = db.cursor()
    rows = cur.execute(query).fetchall()
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in rows]

# 2) Create database with sample data
import os
if os.path.exists("sample.db"):
    os.remove("sample.db")

db = sqlite3.connect("sample.db")
db.executescript(
    """
    CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, price REAL);
    CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT);
    CREATE TABLE orders (
      order_id INTEGER PRIMARY KEY, customer_name TEXT, staff_id INT, product_id INT,
      FOREIGN KEY(staff_id) REFERENCES staff(staff_id),
      FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
    INSERT INTO products(product_name, price) VALUES ('Laptop',799.99),('Keyboard',129.99),('Mouse',29.99);
    INSERT INTO staff(first_name,last_name) VALUES ('Alice','Smith'),('Bob','Johnson');
    INSERT INTO orders(customer_name,staff_id,product_id) VALUES ('David Lee',1,1),('Emily Chen',2,2);
    """
)

# 3) Bind functions for Gemini function calling
client = genai.Client()
model = client.models.get(name="gemini-1.5-flash")

tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="list_tables",
        description=list_tables.__doc__,
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="describe_table",
        description=describe_table.__doc__,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"table": types.Schema(type=types.Type.STRING)},
            required=["table"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_sql",
        description=run_sql.__doc__,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(type=types.Type.STRING)},
            required=["query"],
        ),
    ),
])]

# 4) Minimal agent loop
user_query = "Show me product names and prices from the products table ordered by price desc"
resp = client.models.generate_content(
    model=model.name,
    contents=[user_query],
    config=types.GenerateContentConfig(tools=tools)
)

# If the model wants to call a function, dispatch it
while True:
    calls = resp.function_calls or []
    if not calls:
        print(resp.text)
        break

    call = calls[0]
    fn_name = call.name
    args = dict(call.args) if call.args else {}

    if fn_name == "list_tables":
        fn_result = list_tables()
    elif fn_name == "describe_table":
        fn_result = describe_table(**args)
    elif fn_name == "run_sql":
        fn_result = run_sql(**args)
    else:
        fn_result = {"error": f"unknown function {fn_name}"}

    # Return tool output to the model and continue
    resp = client.models.generate_content(
        model=model.name,
        contents=[types.Content(role="user", parts=[types.Part(text=user_query)]),
                  types.Content(role="tool", parts=[types.Part(function_response=types.FunctionResponse(
                      name=fn_name, response=fn_result
                  ))])],
        config=types.GenerateContentConfig(tools=tools)
    )
```

Key takeaways:

- Define small, typed functions with clear docstrings
- Allow only safe operations (e.g., SELECT-only)
- Bind the functions via schema and drive a tight control loop

## 7. Building an agent with LangGraph (hands-on)

A simple LangGraph agent that uses a ReAct-style loop with a calculator tool:

```python
# pip install langgraph langchain-openai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1) Define a tool
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

# 2) Define agent state
class AgentState(TypedDict):
    input: str
    scratchpad: str
    answer: str

# 3) Model
llm = ChatOpenAI(model="gpt-4o-mini")

# 4) Tool node
tools = [multiply]
tool_node = ToolNode(tools)

# 5) Policy node: decide to call tool or answer
def policy(state: AgentState):
    q = state["input"]
    prompt = f"""
    You are a helpful agent. Decide whether to call the tool 'multiply'.
    If question is about multiplying two numbers, output: TOOL: multiply a=?, b=?
    Otherwise, output a direct answer.
    Question: {q}
    """
    resp = llm.invoke(prompt)
    txt = resp.content if hasattr(resp, "content") else str(resp)
    return {"scratchpad": txt}

# 6) Router: if the LLM requested TOOL:, route to tool node
def router(state: AgentState):
    if "TOOL:" in state.get("scratchpad", ""):
        return "call_tool"
    return "finalize"

# 7) Finalize: produce an answer (very minimal)
def finalize(state: AgentState):
    return {"answer": state.get("scratchpad", "")}

# 8) Build graph
workflow = StateGraph(AgentState)
workflow.add_node("policy", policy)
workflow.add_node("tools", tool_node)
workflow.add_node("finalize", finalize)
workflow.add_edge("policy", router)
workflow.add_edge("tools", "finalize")
workflow.set_entry_point("policy")
workflow.add_conditional_edges("policy", router, {"call_tool": "tools", "finalize": "finalize"})
app = workflow.compile()

# 9) Run
res = app.invoke({"input": "multiply 12.5 by 3"})
print(res["answer"])  # Expect a TOOL call or computed result
```

Notes:

- In real agents, use structured tool schemas, parsing, and rigorous routing
- Add memory (conversation + vector retrieval) and a planner node
- Keep tool execution sandboxed and audited

## 8. Multi-agent patterns

- Specialist agents: split responsibilities (planner, researcher, executor)
- Supervisor/critic loops for quality assurance
- Message passing topologies: hub-and-spoke, round-robin, or DAGs
- When to use: complex workflows that benefit from decomposition

## 9. Evaluation and observability

Measure both quality and operations:

- Task success rate, precision@k (when retrieval is used), factuality
- Latency (p50, p95), cost per request, tool call error rate
- Trace all steps: prompts, tool inputs/outputs, and decisions

## 10. Safety, alignment, and policy

- Input/output filters; jailbreak and prompt-injection defenses
- Action constraints, approvals for sensitive operations
- PII minimization and redaction, audit logging
- Red-team periodically with adversarial prompts

## 11. Production deployment

- Architecture: model gateway, tool services, vector DB, cache, queue, observability
- Patterns: circuit breakers, retries, fallbacks, timeouts, rate limits
- Versioning: prompts, tools, and models; A/B test new releases
- Data flywheel: capture feedback for retraining and prompt refinement

## 12. Hands-on exercises

1. Extend the Gemini function-calling demo to support parameterized filters and safe aggregations.
2. Add a RAG memory to the LangGraph agent and evaluate retrieval quality.
3. Implement a planner-executor split and measure success on multi-step tasks.
4. Add centralized tracing and a dashboard of latency + accuracy.
5. Introduce a human-approval step for sensitive tool calls and test the UX.

## References

- Kaggle Generative AI course (function calling & agents) — distilled in this chapter
- Agents deck and companion notes from the provided PDFs
- LangGraph & LangChain docs; Google Gemini API docs
- Prior chapters: Foundational LLMs, Embeddings & Vector Stores (for RAG and evaluation)
