# tests/unit/test_decision_router.py
"""
Unit tests for the decision router's deterministic routing logic.

The LLM rationale generation is NOT tested here — that's covered by
RAGAS evaluation. These tests only verify the hard-coded thresholds
that the LLM is never allowed to override.
"""

import pytest
from agents.decision_router import _deterministic_route


class TestDeterministicRouting:
    """Verify hard threshold routing — the most safety-critical logic."""

    def test_high_risk_score_rejects(self):
        """Risk >= 0.85 should always reject, regardless of flags."""
        assert _deterministic_route(0.85, []) == 'REJECT'
        assert _deterministic_route(0.90, []) == 'REJECT'
        assert _deterministic_route(1.0, []) == 'REJECT'

    def test_medium_risk_goes_to_human_review(self):
        """Risk between 0.60 and 0.85 should route to human review."""
        assert _deterministic_route(0.60, []) == 'HUMAN_REVIEW'
        assert _deterministic_route(0.70, []) == 'HUMAN_REVIEW'
        assert _deterministic_route(0.84, []) == 'HUMAN_REVIEW'

    def test_low_risk_auto_approves(self):
        """Risk below 0.60 with no critical flags should auto-approve."""
        assert _deterministic_route(0.0, []) == 'AUTO_APPROVE'
        assert _deterministic_route(0.30, []) == 'AUTO_APPROVE'
        assert _deterministic_route(0.59, []) == 'AUTO_APPROVE'

    def test_critical_reg_flags_force_human_review(self):
        """Any HIGH or CRITICAL regulatory flag forces human review."""
        flags = [{'rule': 'Basel III 129(1)', 'severity': 'HIGH', 'excerpt': 'test'}]
        assert _deterministic_route(0.20, flags) == 'HUMAN_REVIEW'

        flags = [{'rule': 'FCA SYSC 6.1', 'severity': 'CRITICAL', 'excerpt': 'test'}]
        assert _deterministic_route(0.10, flags) == 'HUMAN_REVIEW'

    def test_low_severity_flags_dont_block(self):
        """LOW and MEDIUM flags should not override auto-approval."""
        flags = [
            {'rule': 'internal-001', 'severity': 'LOW', 'excerpt': 'minor'},
            {'rule': 'internal-002', 'severity': 'MEDIUM', 'excerpt': 'moderate'},
        ]
        assert _deterministic_route(0.20, flags) == 'AUTO_APPROVE'

    def test_reject_takes_priority_over_flags(self):
        """High risk score should reject even with critical flags."""
        flags = [{'rule': 'test', 'severity': 'CRITICAL', 'excerpt': 'test'}]
        assert _deterministic_route(0.90, flags) == 'REJECT'

    def test_zero_risk_zero_flags(self):
        """Clean document should auto-approve."""
        assert _deterministic_route(0.0, []) == 'AUTO_APPROVE'

    def test_empty_flags_list(self):
        assert _deterministic_route(0.50, []) == 'AUTO_APPROVE'

    def test_none_severity_ignored(self):
        """Flags without severity field should not trigger human review."""
        flags = [{'rule': 'test', 'excerpt': 'test'}]  # no severity key
        assert _deterministic_route(0.20, flags) == 'AUTO_APPROVE'
