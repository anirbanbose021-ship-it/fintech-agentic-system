# evaluation/ragas_eval.py
"""
RAGAS Evaluation Pipeline — deployment gate for the fintech agentic system.

Every deployment candidate MUST pass all four RAGAS metric thresholds
before promotion to production. A faithfulness score below 0.85
immediately blocks deployment and triggers an alert.

Usage:
    # Full evaluation (500 examples) — run on every merge to main
    python evaluation/ragas_eval.py --dataset evaluation/golden_dataset/ --full

    # Smoke evaluation (50 examples) — run on every pull request
    python evaluation/ragas_eval.py --dataset evaluation/golden_dataset/ --smoke

    # Evaluate a specific pipeline run output
    python evaluation/ragas_eval.py --results path/to/pipeline_outputs.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed — install with: pip install ragas datasets")

# ── Production Thresholds ─────────────────────────────────────────────────────
# These are non-negotiable in a regulated fintech environment.
# faithfulness is the primary hallucination guard.

PRODUCTION_THRESHOLDS = {
    "faithfulness":      0.85,  # PRIMARY: hallucination guard — non-negotiable
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.80,
}

SMOKE_SAMPLE_SIZE = 50    # enough to catch regressions in CI
FULL_SAMPLE_SIZE  = 500   # full suite — takes ~8 min on m5.xlarge

sns_client = boto3.client("sns")
ALERT_TOPIC_ARN = "arn:aws:sns:us-east-1:ACCOUNT_ID:ml-deployment-alerts"


def load_golden_dataset(dataset_dir: str, sample_size: Optional[int] = None) -> list:
    """
    Load golden dataset from JSONL files.
    Each record: {input_query, agent_output, retrieved_chunks, golden_answer}
    """
    dataset_path = Path(dataset_dir)
    records = []

    for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                records.append(json.loads(line.strip()))

    if sample_size:
        records = records[:sample_size]

    logger.info(f"Loaded {len(records)} evaluation examples from {dataset_dir}")
    return records


def evaluate_deployment_candidate(
    pipeline_results: list,
    threshold_overrides: Optional[dict] = None,
) -> dict:
    """
    Run RAGAS evaluation against pipeline outputs.

    Args:
        pipeline_results:    List of pipeline output dicts with golden answers
        threshold_overrides: Override specific thresholds (for non-production envs)

    Returns:
        {passed: bool, scores: dict, blocked_by: list}
    """
    if not RAGAS_AVAILABLE:
        raise RuntimeError("RAGAS library not available — install ragas and datasets")

    thresholds = {**PRODUCTION_THRESHOLDS, **(threshold_overrides or {})}

    dataset = Dataset.from_list([
        {
            "question":    r["input_query"],
            "answer":      r["agent_output"],
            "contexts":    r["retrieved_chunks"],
            "ground_truth": r["golden_answer"],
        }
        for r in pipeline_results
    ])

    logger.info(f"Running RAGAS evaluation on {len(dataset)} examples...")

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # mean across all examples — we tried median but it masked outlier failures
    scores_dict = scores.to_pandas().mean().to_dict()

    blocked_by = [
        metric
        for metric, threshold in thresholds.items()
        if scores_dict.get(metric, 0.0) < threshold
    ]

    passed = len(blocked_by) == 0

    result = {
        "passed":     passed,
        "scores":     scores_dict,
        "thresholds": thresholds,
        "blocked_by": blocked_by,
        "sample_size": len(pipeline_results),
    }

    _log_results(result)

    if not passed:
        _send_failure_alert(result)

    return result


def _log_results(result: dict) -> None:
    status = "✓ PASSED" if result["passed"] else "✗ BLOCKED"
    logger.info(f"\n{'='*60}")
    logger.info(f"RAGAS Evaluation Result: {status}")
    logger.info(f"{'='*60}")
    for metric, score in result["scores"].items():
        threshold = result["thresholds"].get(metric, 0.0)
        indicator = "✓" if score >= threshold else "✗"
        logger.info(f"  {indicator} {metric:<25} {score:.4f}  (threshold: {threshold})")
    if result["blocked_by"]:
        logger.warning(f"\nBlocked by: {', '.join(result['blocked_by'])}")
    logger.info(f"{'='*60}\n")


def _send_failure_alert(result: dict) -> None:
    """Notify ML engineering team via SNS when evaluation gate fails."""
    try:
        message = (
            f"RAGAS evaluation gate FAILED — deployment blocked.\n\n"
            f"Blocked metrics: {', '.join(result['blocked_by'])}\n\n"
            f"Scores:\n" +
            "\n".join(
                f"  {m}: {s:.4f} (threshold: {result['thresholds'][m]})"
                for m, s in result["scores"].items()
            )
        )
        sns_client.publish(TopicArn=ALERT_TOPIC_ARN, Subject="[BLOCKED] RAGAS Evaluation Failure", Message=message)
    except Exception as e:
        logger.error(f"Failed to send SNS alert: {e}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation gate")
    parser.add_argument("--dataset",  default="evaluation/golden_dataset/", help="Golden dataset directory")
    parser.add_argument("--results",  help="Path to pipeline outputs JSONL (optional)")
    parser.add_argument("--full",     action="store_true", help="Run full 500-example evaluation")
    parser.add_argument("--smoke",    action="store_true", help="Run 50-example smoke evaluation")
    args = parser.parse_args()

    sample_size = None
    if args.smoke:
        sample_size = SMOKE_SAMPLE_SIZE
    elif not args.full:
        sample_size = SMOKE_SAMPLE_SIZE  # Default to smoke

    if args.results:
        with open(args.results) as f:
            pipeline_results = [json.loads(line) for line in f]
    else:
        pipeline_results = load_golden_dataset(args.dataset, sample_size=sample_size)

    result = evaluate_deployment_candidate(pipeline_results)

    sys.exit(0 if result["passed"] else 1)
