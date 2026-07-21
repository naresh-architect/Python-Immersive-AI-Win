"""
SupportPilot — test ticket suite.

Runs the 8 scenarios from the design doc through the full pipeline and
checks that each one takes the expected resolution path. This is what
you run before the demo to prove the escalation logic actually holds,
including the "same fraud case, rephrased" pair (cases 4 and 5) that
proves the hard rule can't be talked around.
"""
from __future__ import annotations

from pipeline import run_ticket, pretty_print_report

TEST_CASES = [
    {
        "name": "1. Happy path - order status",
        "ticket_id": "T001",
        "customer_id": "CUST001",
        "text": "Hi, can you tell me the status of my recent order? It hasn't arrived yet.",
        "expect": "auto_resolved",
    },
    {
        "name": "2. Refund, in-policy",
        "ticket_id": "T002",
        "customer_id": "CUST003",
        "text": "I'd like a refund for my recent purchase, it doesn't fit well.",
        "expect": "auto_resolved",
    },
    {
        "name": "3. Refund, high-value (CUST002's only order is 6500)",
        "ticket_id": "T003",
        "customer_id": "CUST002",
        "text": "I want a refund on my recent order please.",
        "expect": "escalated",
    },
    {
        "name": "4. Fraud-adjacent, direct phrasing",
        "ticket_id": "T004",
        "customer_id": "CUST005",
        "text": "There's a transaction on my account that I don't recognize at all.",
        "expect": "escalated",
    },
    {
        "name": "5. Same fraud case, softened rephrasing",
        "ticket_id": "T005",
        "customer_id": "CUST005",
        "text": "Hey, quick one - I noticed a small charge on my account, just wanted to double check, I'm not sure it was me.",
        "expect": "escalated",
    },
    {
        "name": "6. Ambiguous / ultra-short ticket -> low confidence",
        "ticket_id": "T006",
        "customer_id": "CUST006",
        "text": "hey issue",
        "expect": "escalated",
    },
    {
        "name": "7. Angry customer, critical urgency",
        "ticket_id": "T007",
        "customer_id": "CUST004",
        "text": "This is absolutely unacceptable, I am furious about how this has been handled, I want a manager.",
        "expect": "escalated",
    },
    {
        "name": "8. No matching KB policy (off-topic, general inquiry)",
        "ticket_id": "T008",
        "customer_id": "CUST011",
        "text": "Do you sell products in other countries besides India?",
        "expect": "auto_resolved",
    },
]


def run_suite() -> None:
    passed, failed = 0, 0
    for case in TEST_CASES:
        report = run_ticket(case["ticket_id"], case["text"], customer_id=case["customer_id"])
        pretty_print_report(report)
        ok = report["resolution_path"] == case["expect"]
        status = "PASS" if ok else "FAIL"
        print(f">>> {case['name']}: expected={case['expect']} actual={report['resolution_path']} [{status}]")
        passed += ok
        failed += not ok

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_suite()
