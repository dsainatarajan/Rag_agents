# -*- coding: utf-8 -*-
"""
3_plan_execute_agent.py

Demonstrates the Plan-and-Execute pattern, in contrast to ReAct (Ex7).

Instead of interleaving one thought/action/observation at a time, this
agent first PLANS the whole task as a numbered list of steps, then EXECUTES
each step one at a time (each step execution is itself a small ReAct-style
tool call), and after each step REPLANS - deciding whether enough
information has been gathered to give a final answer, or whether the
remaining plan needs to be revised.

Graph shape (LangGraph StateGraph):

    planner --> execute_step --> replan --(more steps?)--> execute_step
                                     |
                                     +--(done)--> END


"""

# pip install -U langchain langchain-openai langgraph requests

import json
import re
from typing import TypedDict, List, Tuple, Optional

import requests
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Local vLLM endpoint (OpenAI-compatible /v1) - same pattern as earlier exercises
# ---------------------------------------------------------------------------
LOCAL_BASE_URL = "http://192.168.51.100:8000/v1"
LOCAL_MODEL    = "openai/gpt-oss-120b"
LOCAL_API_KEY  = "not-needed"

llm = ChatOpenAI(
    model=LOCAL_MODEL,
    temperature=0,
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
)

# ---------------------------------------------------------------------------
# Tools + a small ReAct executor agent that carries out ONE step at a time
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather for a named city."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8,
        ).json()
        results = geo.get("results")
        if not results:
            return f"Could not find a location named '{city}'."
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        name = results[0].get("name", city)
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True}, timeout=8,
        ).json()
        current = weather.get("current_weather")
        if not current:
            return f"Weather data unavailable for {name}."
        return f"{name}: {current['temperature']}°C, wind {current['windspeed']} km/h"
    except requests.RequestException as e:
        return f"Weather lookup failed: {e}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live rates."""
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
            timeout=8,
        )
        data = resp.json()
        rate = data.get("rates", {}).get(to_currency.upper())
        if rate is None:
            return f"No rate found for {from_currency.upper()} -> {to_currency.upper()}."
        return f"{amount} {from_currency.upper()} = {round(rate, 2)} {to_currency.upper()}"
    except requests.RequestException as e:
        return f"Currency conversion failed: {e}"


executor_agent = create_agent(
    llm,
    tools=[get_weather, convert_currency],
    system_prompt=(
        "You execute exactly ONE step of a larger plan. Use tools if the "
        "step needs outside data. Reply with a concise result for this step only."
    ),
)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: Optional[str]

# ---------------------------------------------------------------------------
# Helpers - defensive parsing of the LLM's plain-text output
# ---------------------------------------------------------------------------
def parse_numbered_list(text):
    """Extract a numbered/bulleted list of steps from free text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps = []
    for line in lines:
        cleaned = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if cleaned:
            steps.append(cleaned)
    return steps or [text.strip()]


def parse_replan_response(text):
    """
    Replanner replies with either:
      FINAL ANSWER: <answer text>
    or
      REMAINING STEPS:
      1. ...
      2. ...
    """
    if "FINAL ANSWER:" in text:
        return {"response": text.split("FINAL ANSWER:", 1)[1].strip()}
    if "REMAINING STEPS:" in text:
        remainder = text.split("REMAINING STEPS:", 1)[1]
        return {"plan": parse_numbered_list(remainder)}
    # fall back: treat the whole reply as the final answer
    return {"response": text.strip()}

# ---------------------------------------------------------------------------
# Node 1 - Planner: turn the task into a short numbered plan
# ---------------------------------------------------------------------------
planner_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
Break the task below into a short numbered plan (2-5 steps). Each step
should be a single, self-contained action (e.g. "Check the weather in X",
"Convert 500 USD to INR"). Do not solve the task, only plan it.

Task:
{task}

Plan:
"""
)

def plan_step(state: PlanExecuteState):
    response = llm.invoke(planner_prompt.format(task=state["task"]))
    plan = parse_numbered_list(response.content)
    print(f"\n[PLAN]\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)))
    return {"plan": plan, "past_steps": []}

# ---------------------------------------------------------------------------
# Node 2 - Execute: run the FIRST remaining step with the tool-using agent
# ---------------------------------------------------------------------------
def execute_step(state: PlanExecuteState):
    step = state["plan"][0]
    print(f"\n[EXECUTE] {step}")
    prompt = (
        f"Overall task: {state['task']}\n"
        f"Steps already done: {state['past_steps']}\n"
        f"Now execute this step: {step}"
    )
    result = executor_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    output = result["messages"][-1].content
    print(f"[RESULT] {output}")
    return {
        "past_steps": state["past_steps"] + [(step, output)],
        "plan": state["plan"][1:],
    }

# ---------------------------------------------------------------------------
# Node 3 - Replan: decide to continue or produce the final answer
# ---------------------------------------------------------------------------
replan_prompt = PromptTemplate(
    input_variables=["task", "past_steps", "remaining_plan"],
    template="""
Task: {task}

Steps completed so far (step -> result):
{past_steps}

Remaining planned steps: {remaining_plan}

If the completed steps already give enough information to answer the task,
reply with:
FINAL ANSWER: <the answer, written for the user>

Otherwise, reply with the steps still needed as:
REMAINING STEPS:
1. ...
2. ...
"""
)

def replan_step(state: PlanExecuteState):
    past_steps_text = "\n".join(f"- {s} -> {r}" for s, r in state["past_steps"])
    response = llm.invoke(
        replan_prompt.format(
            task=state["task"],
            past_steps=past_steps_text,
            remaining_plan=state["plan"],
        )
    )
    return parse_replan_response(response.content)

def should_continue(state: PlanExecuteState):
    return END if state.get("response") else "execute_step"

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", plan_step)
graph.add_node("execute_step", execute_step)
graph.add_node("replan", replan_step)

graph.set_entry_point("planner")
graph.add_edge("planner", "execute_step")
graph.add_edge("execute_step", "replan")
graph.add_conditional_edges("replan", should_continue, {END: END, "execute_step": "execute_step"})

app = graph.compile()

if __name__ == "__main__":
    task = (
        "Plan a short trip: check the current weather in Chennai and in Paris, "
        "and convert a 500 USD budget to INR for the Chennai leg."
    )
    final_state = app.invoke({"task": task, "plan": [], "past_steps": [], "response": None},
                              config={"recursion_limit": 20})
    print("\n===== Final Answer =====")
    print(final_state["response"])
