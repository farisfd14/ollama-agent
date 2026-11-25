# src/agent.py
# Milestone 2: agent with THREE tools: stocks, math, my notes.
#
# The loop is unchanged (it's always the same loop):
#   send chat+tools -> model asks for a tool -> we run it -> feed result
#   back -> repeat until the model answers with plain text.

import json

import requests
import yfinance as yf

from knowledge import search    # copied from study-bot (Milestone 2 step 1)

OLLAMA_CHAT = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"    #qwen2.5 understands tool calls well

_session = requests.Session()
_session.trust_env = False  # Localhost: no proxy


# ------------- the tools (the hands) ---------------

def get_stock_price(args: dict) -> str:
    ticker = args["ticker"]
    data = yf.Ticker(ticker)
    price = float(data.history(period="5d")["Close"].iloc[-1])
    return f"{ticker} is at {price:.2f}"


def calculator(args: dict) -> str:
    # tiny but honest: only digits and +-*/(). Eval is "safe enough"
    # here because we strip everything else first.
    expr = "".join(c for c in args["expression"] if c in "0123456789+-*/(). ")
    return str(eval(expr))


def search_notes(args: dict) -> str:
    """Search my own documents (the study-bot index)."""
    hits = search(args["query"], top_k=2)
    if not hits:
        return "nothing found in my notes"
    parts = [f"[{chunk['source']}] {chunk['text'][:250]}" for chunk, _ in hits]
    return "\n---\n".join(parts)


# name -> python function. Descriptions live in TOOL_SCHEMA below.
TOOLS = {
    "get_stock_price": get_stock_price,
    "calculator": calculator,
    "search_notes": search_notes,
}

# Ollama's schema format, written out per tool so each can have its
# own argument shape. Keep these descriptions short and concrete -
# the model reads them to decide WHEN to call the tool.
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest stock price. Use for any question about a stock's current price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol like TSLA or AAPL"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for arithmetic the model might get wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '17*6'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search my personal document collection (study notes, dictionaries). Use for questions about my own materials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to search the notes for"}
                },
                "required": ["query"],
            },
        },
    },
]


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Use tools when you need "
                    "real data or math, or my documents. keep answers to 1-2 "
                    "sentences and mention the source when using search_notes."},
        {"role": "user", "content": user_message},
    ]

    # the loop: at most 5 tool-rounds, so a confused model can't spin forever
    for _ in range(5):
        resp = _session.post(OLLAMA_CHAT, json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_SCHEMA,
            "stream": False,
        }, timeout=300)
        msg = resp.json()["message"]

        # no tool requested -> the model is done talking
        if not msg.get("tool_calls"):
            return msg["content"]

        # the model asked for tools -> execute each one, feed results back
        messages.append(msg)
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {})
            if isinstance(args, str):               # some models send JSON as text
                args = json.loads(args)
            print(f"  [tool call] {name}({args})")
            try:
                result = TOOLS[name](args)
            except Exception as error:
                result = f"tool failed: {error}"
            messages.append({"role": "tool",
                             "name": name,
                             "content": result})
    return "(agent hit the 5-round limit - it was probably looping)"


def main():
    print("ollama-agent - type 'quit' to exit")
    while True:
        user = input("you:  ").strip()
        if user.lower() == "quit":
            break
        print(f"agent: {run_agent(user)}")


if __name__ == "__main__":
    main()