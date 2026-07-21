"""
SupportPilot — Pydantic data models.

These are the structured contracts that pass between agents. Every agent
in the pipeline reads and writes these models rather than raw dicts, so
validation happens at every hop (Week 2 discipline, reused throughout).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    DAMAGED_ITEM = "damaged_item"
    PAYMENT_ISSUE = "payment_issue"
    ACCOUNT_ACCESS = "account_access"
    FRAUD_SUSPECTED = "fraud_suspected"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLAINT_ESCALATION = "complaint_escalation"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(str, Enum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    SATISFIED = "satisfied"


class TicketClassification(BaseModel):
    ticket_id: str
    issue_type: IssueType
    urgency: Urgency
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_order_id: Optional[str] = None
    extracted_customer_id: Optional[str] = None
    summary: str


class OrderRecord(BaseModel):
    order_id: str
    item: str
    amount: float
    status: str
    order_date: str


class CustomerContext(BaseModel):
    customer_id: str
    name: str
    email: str
    account_standing: str  # "good" | "flagged" | "vip"
    recent_orders: list[OrderRecord] = []
    prior_ticket_count: int = 0


class RetrievedChunk(BaseModel):
    source_doc: str
    section: str
    text: str
    relevance_score: float


class ValidationResult(BaseModel):
    is_accurate: bool
    is_on_policy: bool
    tone_ok: bool
    issues_found: list[str] = []
    revised_response: Optional[str] = None


class EscalationDecision(BaseModel):
    should_escalate: bool
    reason: str
    escalation_type: Optional[str] = None  # "hard_category" | "low_confidence" | "validation_failed"


class CaseReport(BaseModel):
    ticket_id: str
    classification: TicketClassification
    resolution_path: str  # "auto_resolved" | "escalated"
    final_response: Optional[str] = None
    escalation: Optional[EscalationDecision] = None
    handled_at: datetime = Field(default_factory=datetime.utcnow)
    agent_trace: list[str] = []
