"""
LangGraph ReAct-style agent with an offline fallback policy.

If OPENAI_API_KEY is set, uses ChatOpenAI via LangChain.
Otherwise, a mock policy detects "multiply X by Y" and formats a TOOL call.

Run:
  python examples/agents/langgraph_react_agent.py
"""

import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if USE_OPENAI:
    from langchain_openai import ChatOpenAI


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


class AgentState(TypedDict):
    input: str
    scratchpad: str
    answer: str


def policy(state: AgentState):
    q = state["input"]
    if USE_OPENAI:
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = f"""
        You are a helpful agent. Decide whether to call the tool 'multiply'.
        If question is about multiplying two numbers, output: TOOL: multiply a=?, b=?
        Otherwise, output a direct answer.
        Question: {q}
        """
        resp = llm.invoke(prompt)
        txt = resp.content if hasattr(resp, "content") else str(resp)
        return {"scratchpad": txt}

    # Offline fallback: detect pattern "multiply <num> by <num>"
    m = re.search(r"multiply\s+([0-9.]+)\s+by\s+([0-9.]+)", q, re.I)
    if m:
        a, b = m.groups()
        return {"scratchpad": f"TOOL: multiply a={a}, b={b}"}
    return {"scratchpad": "I can answer without tools."}


def router(state: AgentState):
    return "tools" if "TOOL:" in state.get("scratchpad", "") else "finalize"


def finalize(state: AgentState):
    return {"answer": state.get("scratchpad", "")}


def parse_tool_call(scratchpad: str):
    m = re.search(r"TOOL:\s*multiply\s*a=([0-9.]+),\s*b=([0-9.]+)", scratchpad)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def main():
    tools = [multiply]
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("policy", policy)
    workflow.add_node("tools", tool_node)
    workflow.add_node("finalize", finalize)
    workflow.set_entry_point("policy")
    workflow.add_conditional_edges("policy", router, {"tools": "tools", "finalize": "finalize"})
    workflow.add_edge("tools", "finalize")
    app = workflow.compile()

    question = "multiply 12.5 by 3"
    state = {"input": question, "scratchpad": "", "answer": ""}
    res = app.invoke(state)

    # If we requested a tool in scratchpad, actually call it and print the result
    if "TOOL:" in res.get("answer", ""):
        args = parse_tool_call(res["answer"])
        if args:
            print("Tool result:", multiply.invoke({"a": args[0], "b": args[1]}))
        else:
            print(res["answer"])  # Could not parse
    else:
        print(res["answer"])  # Direct answer


if __name__ == "__main__":
    main()
