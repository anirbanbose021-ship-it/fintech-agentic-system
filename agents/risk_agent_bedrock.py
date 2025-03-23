# agents/risk_agent_bedrock.py
"""
Risk Analysis Agent — AWS Bedrock backend (Claude 3 Sonnet).
Used for REGULATORY and LEGAL documents that do not contain client PII.

Architecture:
  LangGraph → Bedrock Runtime API → Claude 3 Sonnet
"""

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

RISK_SYSTEM_PROMPT = """You are a financial risk assessment specialist with expertise in regulatory and legal analysis.
Analyze the provided document and produce a structured risk assessment.

Scoring guidelines:
- 0.0 – 0.3: Low risk (compliant, no adverse indicators)
- 0.3 – 0.6: Medium risk (minor gaps, requires monitoring)
- 0.6 – 0.8: High risk (significant compliance gaps, human review required)
- 0.8 – 1.0: Critical risk (material non-compliance, recommend rejection or escalation)

Respond ONLY with a JSON object:
{
  "risk_score": <float 0.0-1.0>,
  "risk_factors": ["<factor 1>", "<factor 2>", ...],
  "risk_summary": "<two-sentence summary>"
}"""


def risk_agent_bedrock_node(state: dict) -> dict:
    """
    LangGraph node: risk scoring via AWS Bedrock Claude 3 Sonnet.
    Handles REGULATORY and LEGAL documents.
    """
    chunks = state.get("chunks", [])
    full_text = "\n\n".join([c["text"] for c in chunks[:10]])

    logger.info(
        f"Risk agent (Bedrock) processing document {state['document_id']} "
        f"[type={state.get('document_type')}]"
    )

    response = bedrock_client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        system=[{"text": RISK_SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": f"Assess the risk of this document:\n\n{full_text}"}],
        }],
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )

    raw_text = response["output"]["message"]["content"][0]["text"]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Bedrock risk response parse error: {raw_text}")
        raise ValueError(f"Bedrock returned non-JSON risk response: {e}") from e

    logger.info(
        f"Risk scored: {result['risk_score']:.3f} "
        f"({len(result.get('risk_factors', []))} factors identified)"
    )

    return {
        "risk_score":       result["risk_score"],
        "risk_factors":     result.get("risk_factors", []),
        "risk_agent_model": "bedrock/claude-3-sonnet",
        "audit_events": [{
            "agent":         "risk_analysis",
            "backend":       "bedrock",
            "model":         "claude-3-sonnet",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "risk_score":    result["risk_score"],
            "risk_factors":  result.get("risk_factors", []),
            "input_tokens":  response["usage"]["inputTokens"],
            "output_tokens": response["usage"]["outputTokens"],
        }],
    }
