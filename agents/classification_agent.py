# agents/classification_agent.py
"""
Classification Agent — deployed on AWS Bedrock AgentCore Runtime.
Routes documents into one of four types: KYC | CREDIT | REGULATORY | LEGAL
"""

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

CLASSIFICATION_SYSTEM_PROMPT = """You are a financial document classification specialist.
Classify the provided document excerpt into exactly ONE of these categories:

- KYC: Know Your Customer documents (identity verification, AML checks, customer due diligence)
- CREDIT: Credit assessment documents (loan applications, credit memos, financial statements for lending)
- REGULATORY: Regulatory filings and compliance documents (Basel reports, regulatory submissions, audit reports)
- LEGAL: Legal agreements and contracts (ISDA agreements, NDAs, facility letters, term sheets)

Respond ONLY with a JSON object in this exact format:
{
  "document_type": "<KYC|CREDIT|REGULATORY|LEGAL>",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<one sentence explanation>"
}"""


def classification_node(state: dict) -> dict:
    """
    LangGraph node: classify document type from first 3 chunks.
    Updates state with document_type, classification_confidence, and audit_event.
    """
    chunks = state.get("chunks", [])
    if not chunks:
        raise ValueError("Classification agent received empty chunks — ingestion may have failed")

    # 3 chunks is usually enough for classification — tested on ~200 docs
    # and accuracy didn't improve beyond 3. saves ~40% on token cost
    sample_text = '\n\n'.join([c['text'] for c in chunks[:3]])

    response = bedrock_client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        system=[{"text": CLASSIFICATION_SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": f"Classify this document:\n\n{sample_text}"}]
        }],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},  # Deterministic classification
    )

    raw_text = response["output"]["message"]["content"][0]["text"]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # this happens more than you'd think with sonnet — usually when the
        # doc is very short or mostly tables
        logger.error(f'Classification response parse error: {raw_text}')
        raise ValueError(f'Model returned non-JSON classification response: {e}') from e

    logger.info(
        f"Classified document {state['document_id']} as {result['document_type']} "
        f"(confidence={result['confidence']:.2f})"
    )

    return {
        "document_type":              result["document_type"],
        "classification_confidence":  result["confidence"],
        "audit_events": [{
            "agent":         "classification",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "model":         "claude-3-sonnet",
            "document_type": result["document_type"],
            "confidence":    result["confidence"],
            "rationale":     result.get("rationale"),
            "input_tokens":  response["usage"]["inputTokens"],
            "output_tokens": response["usage"]["outputTokens"],
        }],
    }
