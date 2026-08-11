# -*- coding: utf-8 -*-
"""
2_react_agent.py

Demonstrates the ReAct pattern ("Reason + Act"): the agent alternates between
THOUGHT (reason about what to do next), ACTION (call a tool), and
OBSERVATION (read the tool's result) in a loop, until it has enough
information to give a FINAL ANSWER.

Thought -> Action -> Observation cycle happening under the hood.
"""

# pip install -U langchain langchain-openai requests

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ---------------------------------------------------------------------------
# Local vLLM endpoint (OpenAI-compatible /v1) - same pattern as earlier exercises
# ---------------------------------------------------------------------------
LOCAL_BASE_URL = "http://192.168.51.100:8000/v1"
LOCAL_MODEL    = "openai/gpt-oss-120b"
LOCAL_API_KEY  = "not-needed"


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
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '23 * 4 + 1'."""
    try:
        # restricted eval: digits, operators, parentheses, decimal points only
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Only basic arithmetic (+ - * / and parentheses) is allowed."
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


SYSTEM_PROMPT = (
    "You are a ReAct-style agent. For every user request, reason step by "
    "step about what you still need to find out, call a tool if it helps, "
    "read the result, and keep going until you can give a final answer. "
    "Only answer directly (no tools) if the request needs no outside data."
)


def run_react_agent(question, verbose=True):
    llm = ChatOpenAI(
        model=LOCAL_MODEL,
        temperature=0,
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY,
    )

    agent = create_agent(
        llm,
        tools=[get_weather, calculator],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 12},
    )

    if verbose:
        print("\n--- ReAct trace (Thought -> Action -> Observation) ---")
        for msg in result["messages"]:
            role = msg.__class__.__name__.replace("Message", "")
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    print(f"[{role} - ACTION] call {call['name']}({call['args']})")
            elif role == "Tool":
                print(f"[OBSERVATION] {msg.content}")
            elif getattr(msg, "content", None):
                print(f"[{role}] {msg.content}")

    return result["messages"][-1].content


if __name__ == "__main__":
    question = "What's the current weather in Chennai, and what is 15% of 850?"
    answer = run_react_agent(question)
    print("\n===== Final Answer =====")
    print(answer)
