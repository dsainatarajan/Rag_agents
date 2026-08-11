# -*- coding: utf-8 -*-
"""
trip_planning_agent.py

Tool-calling trip-planning agent (weather + currency conversion + a
human-in-the-loop flight booking gate), running against a LOCAL Gemma model
served via vLLM's OpenAI-compatible endpoint instead of Gemini.
"""

# pip install -U langchain langchain-openai requests

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ---------------------------------------------------------------------------
# Local vLLM endpoint (OpenAI-compatible /v1) 
# ---------------------------------------------------------------------------
LOCAL_BASE_URL = "http://192.168.51.102:8002/v1"
LOCAL_MODEL    = "gemma4-26b"    # must match `curl .../v1/models`
LOCAL_API_KEY  = "not-needed"    # vLLM ignores it, but the SDK needs a value


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


@tool
def book_flight(destination: str, budget_local_currency: float) -> str:
    """Book a flight to a destination with a given budget in the LOCAL
    currency of that destination. This is a SENSITIVE action with a
    real-world side effect -- it pauses for human approval before doing
    anything, no matter how it was triggered."""
    print(f"\n  [GATE] Agent wants to call book_flight(destination={destination!r}, "
          f"budget_local_currency={budget_local_currency})")
    approved = input("  Allow this? (y/n): ").strip().lower() == "y"
    if not approved:
        result = "BLOCKED by human approval gate -- flight not booked."
        print(f"  [GATE] {result}")
        return result
    # No real booking happens -- placeholder action, simulated side effect only.
    result = f"[SIMULATED] Flight to {destination} booked, budget {budget_local_currency}."
    print(f"  [GATE] Approved. {result}")
    return result


SYSTEM_PROMPT = (
    "You are a trip-planning agent. You are given a goal, not a single "
    "question. Reason step by step about what information you still "
    "need, call one or more tools to get it, and continue until the "
    "goal is fully satisfied. Only call book_flight once you have "
    "checked both the weather and the converted budget."
)


def main():
    llm = ChatOpenAI(
        model=LOCAL_MODEL,
        temperature=0,
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY,
    )

    agent = create_agent(
        llm,
        tools=[get_weather, convert_currency, book_flight],
        system_prompt=SYSTEM_PROMPT,
    )

    goal = (
        "Help me plan a trip to Chennai. My budget is 1000 USD. Check the "
        "weather there, convert my budget to INR, and if it seems like a "
        "reasonable trip, go ahead and book the flight."
    )
    print(f"Goal: {goal}")

    # recursion_limit is LangGraph's equivalent of max_steps -- a hard
    # cap on loop iterations so a confused agent can never run forever.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": goal}]},
        config={"recursion_limit": 12},
    )
    final_message = result["messages"][-1]
    print("\nFinal answer:\n")
    print(final_message.content)

if __name__ == "__main__":
    main()
