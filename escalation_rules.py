"""
SupportPilot — Escalation rules.

DELIBERATELY plain Python, no LLM call. The hard-category gate must not
be reachable through prompt rephrasing, so it cannot live inside a model
call — it has to be code the model can't talk its way around.

Two independent gates:
  Gate 1 (hard_escalation): deterministic category/value rules. If this
    fires, escalate — full stop, no confidence score can override it.
  Gate 2 (confidence_escalation): soft, tunable threshold + validation
    outcome.
"""
from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.75
HIGH_VALUE_REFUND_THRESHOLD = 5000.0

HARD_ESCALATE_ISSUE_TYPES = {
    "fraud_suspected",
    "complaint_escalation",
}


def check_hard_escalation(issue_type: str, sentiment: str, urgency: str,
                           order_amount: float = 0.0) -> tuple[bool, str]:
    """Returns (should_escalate, reason). Category/value rules only — no LLM involved."""
    if issue_type in HARD_ESCALATE_ISSUE_TYPES:
        return True, f"issue_type '{issue_type}' is always escalated regardless of confidence"

    if issue_type == "payment_issue":
        return True, "payment issues (duplicate charge, unrecognized transaction) are always escalated to the payments/fraud team"

    if issue_type == "refund_request" and order_amount > HIGH_VALUE_REFUND_THRESHOLD:
        return True, f"refund request for ₹{order_amount:.2f} exceeds the ₹{HIGH_VALUE_REFUND_THRESHOLD:.0f} manual-review threshold"

    if sentiment == "angry" and urgency == "critical":
        return True, "critical urgency combined with angry sentiment requires human handling"

    return False, ""


def check_confidence_escalation(confidence: float, validation_passed: bool) -> tuple[bool, str]:
    """Returns (should_escalate, reason). Soft, tunable gate."""
    if confidence < CONFIDENCE_THRESHOLD:
        return True, f"classifier confidence {confidence:.2f} is below threshold {CONFIDENCE_THRESHOLD}"
    if not validation_passed:
        return True, "draft response failed validation (accuracy, policy, or tone check)"
    return False, ""


def decide_escalation(issue_type: str, sentiment: str, urgency: str, confidence: float,
                       validation_passed: bool, order_amount: float = 0.0) -> dict:
    """Combines both gates into a single decision. Hard gate is checked first and wins."""
    hard, hard_reason = check_hard_escalation(issue_type, sentiment, urgency, order_amount)
    if hard:
        return {"should_escalate": True, "reason": hard_reason, "escalation_type": "hard_category"}

    soft, soft_reason = check_confidence_escalation(confidence, validation_passed)
    if soft:
        etype = "low_confidence" if confidence < CONFIDENCE_THRESHOLD else "validation_failed"
        return {"should_escalate": True, "reason": soft_reason, "escalation_type": etype}

    return {"should_escalate": False, "reason": "passed all checks", "escalation_type": None}


if __name__ == "__main__":
    # Sanity checks — same fraud scenario, phrased differently, must both escalate.
    cases = [
        dict(issue_type="fraud_suspected", sentiment="neutral", urgency="high", confidence=0.9, validation_passed=True, order_amount=0),
        dict(issue_type="fraud_suspected", sentiment="frustrated", urgency="medium", confidence=0.95, validation_passed=True, order_amount=0),
        dict(issue_type="refund_request", sentiment="neutral", urgency="low", confidence=0.9, validation_passed=True, order_amount=6500),
        dict(issue_type="order_status", sentiment="neutral", urgency="low", confidence=0.4, validation_passed=True, order_amount=0),
        dict(issue_type="order_status", sentiment="neutral", urgency="low", confidence=0.9, validation_passed=True, order_amount=0),
    ]
    for c in cases:
        print(decide_escalation(**c))
