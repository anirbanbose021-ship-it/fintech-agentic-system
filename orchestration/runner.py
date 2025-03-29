# orchestration/runner.py
"""
Pipeline execution runner with:
- Human-in-the-loop (HITL) interrupt injection
- Error classification (4xx client errors vs 5xx/429 retriable)
- Dual audit trail: DynamoDB (application-level) + CloudTrail (API-level)
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from orchestration.graph import build_pipeline
from orchestration.state import DocumentPipelineState

logger = logging.getLogger(__name__)

# Error classification: do NOT retry client errors
NON_RETRIABLE_STATUS_CODES = {400, 401, 403, 404, 422}
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class PipelineRunner:
    """
    Dual audit strategy (AWS recommended for regulated workloads):
      - CloudTrail: automatically captures all Bedrock InvokeModel / Converse
        API calls, S3 access, DynamoDB writes — infrastructure-level "who did what"
      - DynamoDB: application-level business audit — risk scores, routing decisions,
        agent rationale, retrieval sources — the stuff CloudTrail can't see

    CloudTrail is configured at the account level via terraform (see infrastructure/terraform/main.tf).
    This runner handles the DynamoDB application audit layer.
    """
    def __init__(self, dynamodb_table: str = "pipeline-audit-log"):
        self.pipeline = build_pipeline()
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(dynamodb_table)

    def run(
        self,
        document_s3_uri: str,
        document_id: str,
        client_id: str,
        hitl_callback: Optional[callable] = None,
    ) -> dict:
        """
        Execute the full document intelligence pipeline.

        Args:
            document_s3_uri: S3 URI of the document to process
            document_id:     Unique document identifier
            client_id:       Client account identifier (used for memory scoping)
            hitl_callback:   Optional async callback for HUMAN_REVIEW routing decisions

        Returns:
            Final pipeline state dict
        """
        initial_state: DocumentPipelineState = {
            "raw_document_s3_uri": document_s3_uri,
            "document_id": document_id,
            "client_id": client_id,
            "chunks": [],
            "document_type": None,
            "classification_confidence": None,
            "risk_score": None,
            "risk_factors": None,
            "risk_agent_model": None,
            "regulatory_flags": None,
            "retrieval_sources": None,
            "routing_decision": None,
            "routing_rationale": None,
            "audit_events": [],
        }

        try:
            final_state = self._execute_with_retry(initial_state)
        except NonRetriableError as e:
            logger.error(f"Non-retriable error for doc {document_id}: {e}")
            self._persist_failed_audit(initial_state, str(e))
            raise

        # HITL intercept: if routed to human review, invoke callback before
        # persisting the final decision
        if final_state.get("routing_decision") == "HUMAN_REVIEW" and hitl_callback:
            logger.info(f"Document {document_id} routed to HUMAN_REVIEW — invoking HITL callback")
            final_state = hitl_callback(final_state)

        self._persist_audit_trail(final_state)
        return final_state

    def _execute_with_retry(self, state: DocumentPipelineState, max_retries: int = 3) -> dict:
        """Execute pipeline with exponential backoff for retriable errors."""
        for attempt in range(max_retries):
            try:
                return self.pipeline.invoke(state)
            except ClientError as e:
                status_code = e.response["Error"].get("HTTPStatusCode", 0)
                if status_code in NON_RETRIABLE_STATUS_CODES:
                    raise NonRetriableError(f"HTTP {status_code}: {e}") from e
                if status_code in RETRIABLE_STATUS_CODES and attempt < max_retries - 1:
                    wait = (2 ** attempt) + (0.1 * attempt)  # exp backoff + small jitter
                    logger.warning(f"Retriable error HTTP {status_code}, retry {attempt + 1} in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Pipeline failed after {max_retries} retries")

    def _persist_audit_trail(self, state: DocumentPipelineState) -> None:
        """
        Persist application-level audit trail to DynamoDB.
        This captures business decisions (risk scores, routing, rationale) that
        CloudTrail doesn't see. CloudTrail handles the infra layer — Bedrock API
        calls, S3 access, IAM activity — automatically via the account-level trail.
        """
        try:
            self.table.put_item(Item={
                "document_id":      state["document_id"],
                "client_id":        state["client_id"],
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "document_type":    state.get("document_type"),
                "risk_score":       str(state.get("risk_score", "")),
                "routing_decision": state.get("routing_decision"),
                "routing_rationale": state.get("routing_rationale"),
                "audit_events":     json.dumps(state.get("audit_events", [])),
                "retrieval_sources": json.dumps(state.get("retrieval_sources", [])),
            })
        except Exception as e:
            logger.error(f"Audit persistence failed for {state['document_id']}: {e}")
            # Never suppress audit failures silently in production
            raise

    def _persist_failed_audit(self, state: DocumentPipelineState, error: str) -> None:
        """Persist failure record for non-retriable errors."""
        try:
            self.table.put_item(Item={
                "document_id": state["document_id"],
                "client_id":   state["client_id"],
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "status":      "FAILED",
                "error":       error,
            })
        except Exception as e:
            logger.error(f"Failed audit persistence error: {e}")


class NonRetriableError(Exception):
    """Raised for 4xx errors that should not be retried."""
    pass


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run document intelligence pipeline")
    parser.add_argument("--doc",         required=True, help="S3 URI of document")
    parser.add_argument("--doc-id",      default="test-doc-001")
    parser.add_argument("--client-id",   default="test-client-001")
    args = parser.parse_args()

    runner = PipelineRunner()
    result = runner.run(
        document_s3_uri=args.doc,
        document_id=args.doc_id,
        client_id=args.client_id,
    )

    print(json.dumps({
        "document_id":      result["document_id"],
        "document_type":    result["document_type"],
        "risk_score":       result["risk_score"],
        "routing_decision": result["routing_decision"],
        "routing_rationale": result["routing_rationale"],
    }, indent=2))
