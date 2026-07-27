# SupportPilot

Multi-agent customer support automation for **ShopStream India** (fictional).
Week 4 capstone — Option A.

Classifies an incoming ticket, looks up the customer's account/orders,
retrieves relevant policy from the knowledge base, drafts a response,
validates it, decides whether to auto-resolve or escalate to a human, and
writes a structured case-closure report — end to end, one function call.

## Quick start

```bash
pip install -r requirements.txt
python3 main.py --init-db          # seed the mock ShopStream database
python3 main.py --test-suite       # run the 8 scenarios from the design doc
python3 main.py --ticket "Where is my order?" --customer CUST001 --json
```

No API key needed to run any of the above — the system defaults to
**mock mode** (see below). This is what makes the test suite fast and
deterministic for your demo prep.

## Two run modes

| Mode | How to enable | What it uses |
|------|---------------|--------------|
| `mock` (default) | nothing — this is the default | Keyword/regex heuristics standing in for the LLM. Zero API calls, zero cost, fully offline. |
| `live` | `export SUPPORTPILOT_MODE=live`<br>`export ANTHROPIC_API_KEY=sk-...` | Real calls to the Anthropic API (`claude-sonnet-5` by default — override with `SUPPORTPILOT_MODEL`) for classification, drafting, and validation. |

Build and test everything in mock mode first. Switch to `live` once the
pipeline shape is solid — you'll immediately see richer, more natural
drafts and classification, at the cost of needing a real API key.

## Project layout

```
supportpilot/
├── models.py             Pydantic schemas (TicketClassification, CaseReport, etc.)
├── database.py           Mock SQLite DB: customers, orders, seed data
├── retrieval.py          TF-IDF knowledge-base retriever over kb/*.md
├── llm_client.py         Mock + live LLM call wrappers (classify/draft/validate)
├── escalation_rules.py   Deterministic hard-category + confidence gates (no LLM)
├── agents.py             One function per agent, thin wrappers around the above
├── pipeline.py           Orchestrates all agents into the full ticket flow
├── test_tickets.py       The 8 scenarios from the design doc, with pass/fail checks
├── main.py               CLI entry point
├── kb/                   5 policy markdown docs (customer-facing + 1 internal)
└── requirements.txt
```

## Why the escalation logic is split into two gates

`escalation_rules.py` deliberately contains **no LLM calls**. The hard
category rules (fraud, duplicate charges, high-value refunds, explicit
"I want a manager" requests) are plain Python `if` statements, checked
*before* the confidence-based gate and given final say. That's what
guarantees a customer can't talk their way around a mandatory escalation
by rephrasing — there's no prompt to manipulate. `test_tickets.py` cases
4 and 5 prove this directly: the same fraud scenario is escalated whether
stated bluntly or hedged politely.

The confidence gate (`CONFIDENCE_THRESHOLD = 0.75` in `escalation_rules.py`)
is the tunable, soft half — adjust it if you find mock-mode confidence
scores don't match how conservative you want the system to be.

## Extending this for the demo

- **Swap the retriever** for real embeddings (OpenAI `text-embedding-3-small`
  or SentenceTransformer) + FAISS/Chroma if you want to show the Week 3
  vector-DB pattern explicitly rather than TF-IDF.
- **Swap the sequential pipeline** in `pipeline.py` for CrewAI or AutoGen if
  your rubric wants a named orchestration framework rather than plain
  function calls — each `agents.py` function maps 1:1 onto an Agent + Task.
- **Add cost/latency tracking** per agent call in live mode (the "Teenager
  Framework" from the course material) by timing each `llm_client` call.
- **Add a lightweight input sanitisation / prompt-injection check** on raw
  ticket text before it reaches the classifier — a preview of the Week 5
  guardrails module, and an easy stretch goal to mention in your debrief.

## Notes on the mock heuristics

`llm_client.py`'s mock mode uses keyword regexes, not a real model — good
enough to prove the *pipeline shape and escalation logic* work, but it will
occasionally misclassify phrasing it hasn't seen. That's expected and worth
saying explicitly in your demo: mock mode validates the architecture, live
mode is what you'd actually ship. Don't over-tune the regexes to make mock
mode look smarter than it is; it's a stand-in, not the deliverable.
