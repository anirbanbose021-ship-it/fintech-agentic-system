# tests/unit/test_graph_routing.py
"""
Unit tests for the LangGraph conditional routing logic.
Verifies that document types are routed to the correct model backend.
"""

import pytest
from orchestration.graph import route_to_risk_backend


class TestModelBackendRouting:
    """KYC/CREDIT -> vLLM (on-prem), everything else -> Bedrock."""

    def test_kyc_routes_to_vllm(self):
        state = {'document_type': 'KYC'}
        assert route_to_risk_backend(state) == 'risk_agent_vllm'

    def test_credit_routes_to_vllm(self):
        state = {'document_type': 'CREDIT'}
        assert route_to_risk_backend(state) == 'risk_agent_vllm'

    def test_regulatory_routes_to_bedrock(self):
        state = {'document_type': 'REGULATORY'}
        assert route_to_risk_backend(state) == 'risk_agent_bedrock'

    def test_legal_routes_to_bedrock(self):
        state = {'document_type': 'LEGAL'}
        assert route_to_risk_backend(state) == 'risk_agent_bedrock'

    def test_unknown_type_routes_to_bedrock(self):
        """Unknown types default to bedrock — safer than on-prem for unknowns."""
        state = {'document_type': 'UNKNOWN'}
        assert route_to_risk_backend(state) == 'risk_agent_bedrock'

    def test_none_type_routes_to_bedrock(self):
        state = {'document_type': None}
        assert route_to_risk_backend(state) == 'risk_agent_bedrock'

    def test_missing_type_routes_to_bedrock(self):
        state = {}
        assert route_to_risk_backend(state) == 'risk_agent_bedrock'
