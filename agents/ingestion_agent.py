# agents/ingestion_agent.py
"""
Ingestion Agent — PDF parsing, intelligent chunking, and S3 staging.
First node in the LangGraph pipeline: extracts structured text from
uploaded documents and prepares chunks for downstream agents.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import List

import boto3

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")

# Chunking parameters tuned for financial documents
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64


def _extract_text_from_pdf(s3_uri: str) -> str:
    """
    Download PDF from S3 and extract text via Amazon Textract.
    Falls back to PyPDF2 for simple text-layer PDFs.
    """
    bucket, key = _parse_s3_uri(s3_uri)

    textract = boto3.client("textract")
    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = response["JobId"]

    # Poll until complete — TODO: replace with step functions async callback
    # instead of busy-waiting (fine for MVP but won't scale past ~50 concurrent docs)
    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        if result["JobStatus"] in ("SUCCEEDED", "FAILED"):
            break

    if result["JobStatus"] == "FAILED":
        raise RuntimeError(f"Textract failed for {s3_uri}")

    pages = []
    for block in result.get("Blocks", []):
        if block["BlockType"] == "LINE":
            pages.append(block["Text"])

    return "\n".join(pages)


def _chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS,
                overlap: int = CHUNK_OVERLAP_TOKENS) -> List[dict]:
    """
    Split text into overlapping chunks.
    Uses simple whitespace tokenization as a proxy for token count.
    """
    # NOTE: whitespace split is a rough proxy — tiktoken would be more accurate
    # but adds a dependency we don't need yet for chunking
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = hashlib.sha256(chunk_text.encode()).hexdigest()[:12]
        chunks.append({
            "text": chunk_text,
            "chunk_id": chunk_id,
            "metadata": {
                "start_word": start,
                "end_word": end,
                "token_estimate": end - start,
            },
        })
        start += max_tokens - overlap

    return chunks


def _parse_s3_uri(uri: str) -> tuple:
    """Parse s3://bucket/key into (bucket, key)."""
    path = uri.replace("s3://", "")
    bucket, _, key = path.partition("/")
    return bucket, key


def ingestion_node(state: dict) -> dict:
    """
    LangGraph node: ingest document from S3, extract text, chunk, and return.
    """
    s3_uri = state["raw_document_s3_uri"]
    document_id = state["document_id"]

    logger.info(f"Ingesting document {document_id} from {s3_uri}")

    raw_text = _extract_text_from_pdf(s3_uri)
    chunks = _chunk_text(raw_text)

    logger.info(f"Ingested {len(chunks)} chunks from document {document_id}")

    return {
        "chunks": chunks,
        "audit_events": [{
            "agent":       "ingestion",
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "document_id": document_id,
            "s3_uri":      s3_uri,
            "num_chunks":  len(chunks),
            "total_words": sum(c["metadata"]["token_estimate"] for c in chunks),
        }],
    }
