"""
SupportPilot — Pipeline orchestration.

Wires the agents together in the order the design doc specifies:
classify -> account lookup + knowledge retrieval -> draft -> validate
-> escalation decision -> case closure report.

This is a plain sequential Python pipeline rather than a framework
(CrewAI/AutoGen) so it has zero extra dependencies and is easy to trace
step by step in a demo. Swapping this for CrewAI/AutoGen later is
straightforward: each function below maps 1:1 onto an Agent + Task.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import agents
import escalation_rules

try:
    from models import TicketClassification
    _PYDANTIC_AVAILABLE = True
except ImportError:  # pydantic not installed in this environment
    _PYDANTIC_AVAILABLE = False


def run_ticket(ticket_id: str, ticket_text: str, customer_id: str | None = None) -> dict:
    trace: list[str] = []

    # 1. Classify
    classification = agents.classifier_agent(ticket_id, ticket_text)
    trace.append("classifier_agent")

    # Structured-intake validation gate (Week 2 discipline): the pipeline must not
    # silently proceed on an invalid or incomplete extraction. If pydantic is
    # installed, this actually enforces the schema; if not, it's a no-op so the
    # rest of the pipeline still runs in this sandbox.
    if _PYDANTIC_AVAILABLE:
        classification = TicketClassification(**classification).model_dump()
        trace.append("pydantic_validation")

    # Prefer an explicitly-passed customer_id, fall back to one the classifier extracted
    resolved_customer_id = customer_id or classification.get("extracted_customer_id")

    # 2. Account lookup + 3. Knowledge retrieval (independent, both feed drafting)
    customer_ctx = agents.account_lookup_agent(resolved_customer_id)
    trace.append("account_lookup_agent")

    kb_query = f"{classification['issue_type']} {classification['summary']}"
    kb_chunks = agents.knowledge_retrieval_agent(kb_query)
    trace.append("knowledge_retrieval_agent")

    # 4. Draft
    draft = agents.drafting_agent(classification, customer_ctx, kb_chunks)
    trace.append("drafting_agent")

    # 5. Validate
    validation = agents.validation_agent(draft, kb_chunks, classification)
    trace.append("validation_agent")

    # 6. Escalation decision (deterministic — see escalation_rules.py)
    order_amount = agents.get_relevant_order_amount(customer_ctx, classification)
    validation_passed = validation["is_accurate"] and validation["is_on_policy"] and validation["tone_ok"]
    decision = escalation_rules.decide_escalation(
        issue_type=classification["issue_type"],
        sentiment=classification["sentiment"],
        urgency=classification["urgency"],
        confidence=classification["confidence"],
        validation_passed=validation_passed,
        order_amount=order_amount,
    )
    trace.append("escalation_agent")

    # 7. Resolve or escalate
    resolution_path = "escalated" if decision["should_escalate"] else "auto_resolved"
    final_response = None if decision["should_escalate"] else draft

    # 8. Case closure report
    report = {
        "ticket_id": ticket_id,
        "classification": classification,
        "customer_context_found": customer_ctx is not None,
        "kb_chunks_used": [f"{c['source_doc']}#{c['section']}" for c in kb_chunks],
        "draft_response": draft,
        "validation": validation,
        "order_amount_considered": order_amount,
        "resolution_path": resolution_path,
        "final_response": final_response,
        "escalation": decision,
        "handled_at": datetime.now(timezone.utc).isoformat(),
        "agent_trace": trace + ["case_closure_agent"],
    }
    return report


def pretty_print_report(report: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Ticket: {report['ticket_id']}  |  Path: {report['resolution_path'].upper()}")
    print(f"{'='*70}")
    c = report["classification"]
    print(f"Issue type: {c['issue_type']} | Urgency: {c['urgency']} | Sentiment: {c['sentiment']} | Confidence: {c['confidence']:.2f}")
    print(f"KB chunks used: {report['kb_chunks_used'] or 'none'}")
    if report["resolution_path"] == "auto_resolved":
        print(f"Response: {report['final_response']}")
    else:
        esc = report["escalation"]
        print(f"ESCALATED — type: {esc['escalation_type']} | reason: {esc['reason']}")
    print(f"Agent trace: {' -> '.join(report['agent_trace'])}")


if __name__ == "__main__":
    report = run_ticket("T-DEMO-001", "Where is my order? It's been 6 days and tracking hasn't updated.", customer_id="CUST001")
    pretty_print_report(report)
    print(json.dumps(report, indent=2, default=str))
