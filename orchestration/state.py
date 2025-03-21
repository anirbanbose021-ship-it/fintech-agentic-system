# orchestration/state.py
import operator
from typing import TypedDict, List, Optional, Annotated


class DocumentPipelineState(TypedDict):
    # ── Ingestion ────────────────────────────────────────────────────────────
    raw_document_s3_uri: str
    document_id: str
    client_id: str
    chunks: List[dict]                    # [{text, metadata, chunk_id}]

    # ── Classification ───────────────────────────────────────────────────────
    document_type: Optional[str]          # KYC | CREDIT | REGULATORY | LEGAL
    classification_confidence: Optional[float]

    # ── Risk Analysis ────────────────────────────────────────────────────────
    risk_score: Optional[float]           # 0.0 – 1.0
    risk_factors: Optional[List[str]]
    risk_agent_model: Optional[str]       # tracks which backend was used

    # ── Regulatory Cross-Reference ───────────────────────────────────────────
    regulatory_flags: Optional[List[dict]]  # [{rule, severity, excerpt}]
    retrieval_sources: Optional[List[str]]

    # ── Decision Routing ─────────────────────────────────────────────────────
    routing_decision: Optional[str]       # AUTO_APPROVE | HUMAN_REVIEW | REJECT
    routing_rationale: Optional[str]

    # ── Audit Trail (accumulates across all agents) ──────────────────────────
    audit_events: Annotated[List[dict], operator.add]
