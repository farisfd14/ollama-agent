# ollama-agent

A local LLM agent (qwen2.5:3b via Ollama) that can USE tools, not just
talk. Terminal-based, 100% local, no API keys.

## What it can do

| you say | the agent does |
| --- | --- |
| "How's TSLA?" | calls `get_stokc_price` itself, reads the real number, answers |
| "What is 17 times 6?" | calls `calculator` (small models hallucinate math - the tool doesn't |
| "什么是卷积?" | calls `search_notes` on my own document collection, answers with the source |

Watch the `[tool call]` lines in the terminal - that's the model deciding which hand to use.

## The agent loop (the whole architecture, honestly)

1. send chat + tool schemas to Ollama
2. model replies with either text (done) or a tool request
3. WE execute the requested function and append the result as a "tool" message
4. repeat (max 5 rounds, so a confused model can't loop forever)

The model never runs code - it can on;y *ask*. The loop is the
gatekeeper. Every production agent system is this loop with more tools
and more safety rails.

## Tools

- `get_stock_price(tiker)` - yfinance (lesson from stock-lab: rates can limit; this is a demo)
- `calculator(expression)` - real math instead of remembered math
- `search_notes(query)` - RAG over my own documents, reuses study bot's knowledge.py (vector index, cosine similarity)

## Setup

- `pip install -r requirements.txt`
- `ollama pull qwen2.5:3b`
- Copy your documents (PDFs) into `data/` and build the index once
- `python src/knowledge.py"test question"`
- Then: `python src/agent.py`

## Design decisions worth copying

- `pick_answer` patatern: never hardcode Ollama's JSON field names
- `trust_env = False` for localhost, but NEVER for external APIs (Telegram needs the proxy in China - opposite rules per destination)
- tool results fed back as `role: "tool"` messages - the model reads the REAL result, not its guess
- failures inside a tool return "tool failed: ..." as the result - the model sees the failure and can react, instead of the agent crashing

## Measured by local-model-bench

The model doing the tool-calling here is the same one benched in my local-model-bench repo - quality 0.77, ~7.9s/question on my CPU. The repos reference each other's data on purpose.