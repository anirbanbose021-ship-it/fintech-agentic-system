# agents/regulatory_agent.py
"""
Regulatory Cross-Reference Agent — hybrid RAG over regulatory corpus.
Retrieves relevant regulations from OpenSearch Serverless and cross-references
against document content to flag compliance gaps.
"""

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
import os

# TODO: move this to env var before prod deploy
opensearch_endpoint = os.environ.get(
    'OPENSEARCH_ENDPOINT', 'https://COLLECTION_ID.us-east-1.aoss.amazonaws.com'
)

REGULATORY_SYSTEM_PROMPT = """You are a regulatory compliance specialist in financial services.
Given a document excerpt and retrieved regulatory references, identify any compliance gaps.

For each gap found, provide:
- rule: The specific regulation or rule reference
- severity: LOW | MEDIUM | HIGH | CRITICAL
- excerpt: The relevant text from the document that triggers this flag

Respond ONLY with a JSON object:
{
  "regulatory_flags": [
    {"rule": "<regulation reference>", "severity": "<level>", "excerpt": "<text>"},
    ...
  ],
  "summary": "<one-sentence compliance summary>"
}

If no compliance gaps are found, return an empty flags array."""


def _retrieve_regulatory_context(query_text: str, top_k: int = 5) -> list:
    """
    Hybrid retrieval: semantic search via Amazon Titan Embed v2
    against OpenSearch Serverless regulatory corpus.
    """
    # Generate embedding via Bedrock Titan Embed v2
    embed_response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": query_text[:2048]}),
    )
    embedding = json.loads(embed_response["body"].read())["embedding"]

    # lazy import — opensearch-py is heavy and not needed by other agents
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    session = boto3.Session()
    credentials = session.get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        "us-east-1",
        "aoss",
        session_token=credentials.token,
    )

    client = OpenSearch(
        hosts=[{"host": opensearch_endpoint.replace("https://", ""), "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        connection_class=RequestsHttpConnection,
    )

    response = client.search(
        index="regulatory-corpus",
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                    }
                }
            },
        },
    )

    return [
        {
            "text": hit["_source"]["text"],
            "regulation": hit["_source"].get("regulation_id", "unknown"),
            "score": hit["_score"],
        }
        for hit in response["hits"]["hits"]
    ]


def regulatory_crossref_node(state: dict) -> dict:
    """
    LangGraph node: cross-reference document against regulatory corpus.
    Uses hybrid RAG to retrieve relevant regulations and flag compliance gaps.
    """
    chunks = state.get("chunks", [])
    document_type = state.get("document_type", "UNKNOWN")

    # Build query from risk factors + document content
    risk_factors = state.get("risk_factors", [])
    query_text = (
        f"Document type: {document_type}\n"
        f"Risk factors: {', '.join(risk_factors)}\n\n"
        + "\n\n".join([c["text"] for c in chunks[:5]])
    )

    logger.info(f"Regulatory cross-ref for document {state['document_id']}")

    # Retrieve relevant regulatory context
    reg_contexts = _retrieve_regulatory_context(query_text)
    context_text = "\n\n".join(
        f"[{ctx['regulation']}]: {ctx['text']}" for ctx in reg_contexts
    )

    # LLM cross-reference analysis
    response = bedrock_client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        system=[{"text": REGULATORY_SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{
                "text": (
                    f"Document excerpt:\n{query_text}\n\n"
                    f"Regulatory references:\n{context_text}\n\n"
                    f"Identify any compliance gaps."
                )
            }],
        }],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
    )

    raw_text = response["output"]["message"]["content"][0]["text"]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Regulatory response parse error: {raw_text}")
        raise ValueError(f"Model returned non-JSON regulatory response: {e}") from e

    flags = result.get("regulatory_flags", [])
    sources = [ctx["regulation"] for ctx in reg_contexts]

    logger.info(f"Found {len(flags)} regulatory flags for document {state['document_id']}")

    return {
        "regulatory_flags":  flags,
        "retrieval_sources": sources,
        "audit_events": [{
            "agent":            "regulatory_crossref",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "model":            "claude-3-sonnet",
            "num_flags":        len(flags),
            "retrieval_sources": sources,
            "input_tokens":     response["usage"]["inputTokens"],
            "output_tokens":    response["usage"]["outputTokens"],
        }],
    }
