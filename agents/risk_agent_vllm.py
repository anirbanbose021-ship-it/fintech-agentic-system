# agents/risk_agent_vllm.py
"""
Risk Analysis Agent — Self-hosted Mistral 7B via vLLM + LiteLLM proxy.
Used for KYC and CREDIT documents containing client PII that cannot
leave the organization's private VPC perimeter.

Architecture:
  LangGraph → LiteLLM proxy (OpenAI-compatible) → vLLM → Mistral 7B (fine-tuned)
"""

import json
import logging
import os
from datetime import datetime, timezone

from openai import OpenAI

logger = logging.getLogger(__name__)

# LiteLLM proxy presents an OpenAI-compatible endpoint
# This means this agent code is IDENTICAL to risk_agent_bedrock.py
# — only the client configuration differs
litellm_client = OpenAI(
    base_url=os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000"),
    api_key=os.environ.get("LITELLM_API_KEY", "sk-local"),
)

VLLM_MODEL = os.environ.get("VLLM_MODEL_ID", "mistral-7b-finetuned-credit")

RISK_SYSTEM_PROMPT = """You are a financial risk assessment specialist with expertise in credit analysis.
Analyze the provided document and produce a structured risk assessment.

Scoring guidelines:
- 0.0 – 0.3: Low risk (strong financials, clean KYC, no adverse indicators)
- 0.3 – 0.6: Medium risk (some adverse indicators, requires monitoring)
- 0.6 – 0.8: High risk (significant adverse indicators, human review required)
- 0.8 – 1.0: Critical risk (recommend rejection or escalation)

Respond ONLY with a JSON object:
{
  "risk_score": <float 0.0-1.0>,
  "risk_factors": ["<factor 1>", "<factor 2>", ...],
  "risk_summary": "<two-sentence summary>"
}"""


def risk_agent_vllm_node(state: dict) -> dict:
    """
    LangGraph node: risk scoring via self-hosted Mistral 7B (vLLM).
    Handles sensitive KYC/CREDIT documents that must stay on-premises.
    """
    chunks = state.get("chunks", [])
    # For risk scoring, use broader document context than classification
    full_text = "\n\n".join([c["text"] for c in chunks[:10]])

    logger.info(
        f"Risk agent (vLLM) processing document {state['document_id']} "
        f"[type={state.get('document_type')}]"
    )

    # logger.debug(f"vLLM payload size: {len(full_text)} chars")
    completion = litellm_client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": RISK_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Assess the risk of this document:\n\n{full_text}"},
        ],
        max_tokens=512,
        temperature=0.1,
    )

    raw_text = completion.choices[0].message.content

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"vLLM risk response parse error: {raw_text}")
        raise ValueError(f"vLLM returned non-JSON risk response: {e}") from e

    logger.info(
        f"Risk scored: {result['risk_score']:.3f} "
        f"({len(result.get('risk_factors', []))} factors identified)"
    )

    return {
        "risk_score":       result["risk_score"],
        "risk_factors":     result.get("risk_factors", []),
        "risk_agent_model": f"vllm/{VLLM_MODEL}",
        "audit_events": [{
            "agent":         "risk_analysis",
            "backend":       "vllm_on_prem",
            "model":         VLLM_MODEL,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "risk_score":    result["risk_score"],
            "risk_factors":  result.get("risk_factors", []),
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }],
    }
