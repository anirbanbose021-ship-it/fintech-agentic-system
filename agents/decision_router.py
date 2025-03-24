# agents/decision_router.py
"""
Decision Router Agent — aggregates signals from upstream agents and
produces a final routing decision: AUTO_APPROVE | HUMAN_REVIEW | REJECT.

Deterministic thresholds override the LLM for safety-critical decisions,
with the LLM providing rationale only.
"""

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Hard thresholds — the LLM NEVER overrides these.
# Learned this the hard way: letting the model "suggest" routing
# led to 3 auto-approvals on high-risk docs during load testing
REJECT_RISK_THRESHOLD = 0.85
HUMAN_REVIEW_RISK_THRESHOLD = 0.60
CRITICAL_REG_SEVERITY = {'HIGH', 'CRITICAL'}

RATIONALE_SYSTEM_PROMPT = """You are a decision rationale generator for a financial document processing pipeline.
Given the document analysis signals below, write a concise (2-3 sentence) rationale
for the routing decision that has already been made. Do NOT change the decision.

Respond ONLY with a JSON object:
{
  "rationale": "<2-3 sentence rationale>"
}"""


def _deterministic_route(risk_score: float, regulatory_flags: list) -> str:
    """
    Apply hard-coded business rules. The LLM never overrides these.
    """
    if risk_score >= REJECT_RISK_THRESHOLD:
        return "REJECT"

    critical_flags = [
        f for f in regulatory_flags
        if f.get("severity") in CRITICAL_REG_SEVERITY
    ]
    if critical_flags:
        return "HUMAN_REVIEW"

    if risk_score >= HUMAN_REVIEW_RISK_THRESHOLD:
        return "HUMAN_REVIEW"

    return "AUTO_APPROVE"


def decision_router_node(state: dict) -> dict:
    """
    LangGraph node: aggregate all signals and route the document.
    """
    risk_score = state.get("risk_score", 0.0) or 0.0
    regulatory_flags = state.get("regulatory_flags", []) or []

    decision = _deterministic_route(risk_score, regulatory_flags)

    logger.info(
        f"Document {state['document_id']} routed to {decision} "
        f"(risk={risk_score:.3f}, flags={len(regulatory_flags)})"
    )

    # Generate human-readable rationale via LLM
    signals_summary = (
        f"Document type: {state.get('document_type')}\n"
        f"Risk score: {risk_score:.3f}\n"
        f"Risk factors: {state.get('risk_factors', [])}\n"
        f"Regulatory flags: {len(regulatory_flags)}\n"
        f"Decision: {decision}"
    )

    response = bedrock_client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",  # Haiku for cost
        system=[{"text": RATIONALE_SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": signals_summary}],
        }],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )

    raw_text = response["output"]["message"]["content"][0]["text"]
    try:
        rationale = json.loads(raw_text).get("rationale", "")
    except json.JSONDecodeError:
        # haiku occasionally returns bare text instead of json, not a big deal
        rationale = raw_text

    return {
        "routing_decision":  decision,
        "routing_rationale": rationale,
        "audit_events": [{
            "agent":             "decision_router",
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "model":             "claude-3-haiku",
            "routing_decision":  decision,
            "risk_score":        risk_score,
            "num_reg_flags":     len(regulatory_flags),
            "input_tokens":      response["usage"]["inputTokens"],
            "output_tokens":     response["usage"]["outputTokens"],
        }],
    }
