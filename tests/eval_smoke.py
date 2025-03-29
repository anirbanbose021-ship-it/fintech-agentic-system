# tests/eval_smoke.py
"""
50-example RAGAS smoke test — runs on every pull request.

This is NOT a replacement for the full 500-example evaluation gate.
It catches obvious regressions quickly (~90 seconds vs ~8 minutes for full).

Usage:
    pytest tests/eval_smoke.py -v
    # or directly:
    python tests/eval_smoke.py
"""

import sys
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ragas_eval import load_golden_dataset, evaluate_deployment_candidate, SMOKE_SAMPLE_SIZE
from evaluation.thresholds import get_thresholds


def test_smoke_evaluation():
    """
    Run 50-example smoke evaluation against staging thresholds.
    PR gate — must pass before merge.
    """
    dataset_dir = Path(__file__).parent.parent / 'evaluation' / 'golden_dataset'

    if not dataset_dir.exists() or not list(dataset_dir.glob('*.jsonl')):
        print('SKIP: golden dataset not found — run full eval manually')
        return

    results = load_golden_dataset(str(dataset_dir), sample_size=SMOKE_SAMPLE_SIZE)

    if len(results) < 5:
        print(f'SKIP: only {len(results)} examples in golden dataset (need >= 5)')
        return

    # use staging thresholds for PR smoke tests — production thresholds
    # are enforced on merge to main
    staging_thresholds = get_thresholds('staging')
    outcome = evaluate_deployment_candidate(results, threshold_overrides=staging_thresholds)

    assert outcome['passed'], (
        f"Smoke evaluation FAILED. Blocked by: {outcome['blocked_by']}. "
        f"Scores: {outcome['scores']}"
    )


if __name__ == '__main__':
    test_smoke_evaluation()
    print('Smoke evaluation passed.')
