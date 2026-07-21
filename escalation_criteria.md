# Internal Escalation Criteria (Not Customer-Facing)

This document grounds the Validation and Escalation agents. It is never shown to a customer.

## Always Escalate (Hard Rule — No Exceptions)
- Any suspected fraud or unrecognized transaction claim
- Any refund request where order value exceeds ₹5000
- Duplicate charge complaints
- Any request explicitly asking for a manager, legal action, or threatening public/media escalation
- Account compromise / unauthorized access reports

## Escalate on Low Confidence
- Classifier confidence below 0.75
- Draft response fails validation (inaccurate, off-policy, or wrong tone)
- Ticket content is ambiguous enough that issue_type cannot be determined with reasonable certainty

## Do Not Escalate (safe to auto-resolve when validation passes)
- Order status inquiries
- Standard refund requests within policy and under ₹5000
- Damaged item reports within the 48-hour window
- General FAQ questions about policy
