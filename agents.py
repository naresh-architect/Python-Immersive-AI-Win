"""
SupportPilot — Agent functions.

Each function is one agent's responsibility, kept small and single-purpose
so the pipeline in pipeline.py can trace exactly which agent produced what.
"""
from __future__ import annotations

import database
import llm_client
from retrieval import KnowledgeRetriever

_retriever = KnowledgeRetriever()


def classifier_agent(ticket_id: str, ticket_text: str) -> dict:
    return llm_client.classify_ticket(ticket_id, ticket_text)


def account_lookup_agent(customer_id: str | None) -> dict | None:
    if not customer_id:
        return None
    customer = database.get_customer(customer_id)
    if not customer:
        return None
    orders = database.get_orders(customer_id)
    customer["recent_orders"] = orders
    customer["prior_ticket_count"] = database.get_prior_ticket_count(customer_id)
    return customer


def knowledge_retrieval_agent(query: str, top_k: int = 3) -> list[dict]:
    return _retriever.retrieve(query, top_k=top_k)


def drafting_agent(classification: dict, customer_ctx: dict | None, kb_chunks: list[dict]) -> str:
    return llm_client.draft_response(classification, customer_ctx, kb_chunks)


def validation_agent(draft: str, kb_chunks: list[dict], classification: dict) -> dict:
    return llm_client.validate_response(draft, kb_chunks, classification)


def get_relevant_order_amount(customer_ctx: dict | None, classification: dict) -> float:
    """Best-effort lookup of the order amount relevant to a refund/payment ticket,
    used by the escalation gate's high-value-refund rule."""
    if not customer_ctx:
        return 0.0
    order_id = classification.get("extracted_order_id")
    orders = customer_ctx.get("recent_orders", [])
    if order_id:
        for o in orders:
            if o["order_id"].upper() == order_id.upper():
                return float(o["amount"])
    # fall back to the most recent order if no specific order was referenced
    return float(orders[0]["amount"]) if orders else 0.0
