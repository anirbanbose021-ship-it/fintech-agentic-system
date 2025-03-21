# orchestration/graph.py
"""
LangGraph multi-agent orchestration graph for fintech document intelligence.
Five specialized agents connected via explicit state transitions.
"""

from langgraph.graph import StateGraph, END
from orchestration.state import DocumentPipelineState
from agents.ingestion_agent import ingestion_node
from agents.classification_agent import classification_node
from agents.risk_agent_bedrock import risk_agent_bedrock_node
from agents.risk_agent_vllm import risk_agent_vllm_node
from agents.regulatory_agent import regulatory_crossref_node
from agents.decision_router import decision_router_node


SENSITIVE_DOCUMENT_TYPES = {"KYC", "CREDIT"}


def route_to_risk_backend(state: DocumentPipelineState) -> str:
    """
    Routing logic: sensitive document types (KYC, CREDIT) containing client PII
    are routed to the self-hosted vLLM endpoint inside the private VPC.
    All other document types use the Bedrock-hosted risk agent.
    """
    if state.get("document_type") in SENSITIVE_DOCUMENT_TYPES:
        return "risk_agent_vllm"
    return "risk_agent_bedrock"


def build_pipeline() -> StateGraph:
    """
    Compile the document intelligence pipeline.
    Returns a runnable LangGraph graph.

    Graph topology:
        ingestion → classification → [conditional] → risk_agent_vllm ──┐
                                                   → risk_agent_bedrock ┤
                                                                        ↓
                                                             regulatory_crossref
                                                                        ↓
                                                             decision_router → END
    """
    graph = StateGraph(DocumentPipelineState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("ingestion",           ingestion_node)
    graph.add_node("classification",      classification_node)
    graph.add_node("risk_agent_vllm",     risk_agent_vllm_node)
    graph.add_node("risk_agent_bedrock",  risk_agent_bedrock_node)
    graph.add_node("regulatory_crossref", regulatory_crossref_node)
    graph.add_node("decision_router",     decision_router_node)

    # ── Wire edges ───────────────────────────────────────────────────────────
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "classification")

    # Conditional routing: KYC/CREDIT → vLLM, others → Bedrock
    graph.add_conditional_edges(
        "classification",
        route_to_risk_backend,
        {
            "risk_agent_vllm":    "risk_agent_vllm",
            "risk_agent_bedrock": "risk_agent_bedrock",
        },
    )

    graph.add_edge("risk_agent_vllm",    "regulatory_crossref")
    graph.add_edge("risk_agent_bedrock", "regulatory_crossref")
    graph.add_edge("regulatory_crossref", "decision_router")
    graph.add_edge("decision_router",     END)

    return graph.compile()
