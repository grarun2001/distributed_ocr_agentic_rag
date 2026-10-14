"""LangGraph multi-agent workflow compilation and execution graph."""

from langgraph.graph import StateGraph, END
from app.agents.nodes import (
    retrieve_and_evaluate_node,
    web_search_node,
    generate_and_verify_answer_node
)
from app.agents.routing import route_evidence_check
from app.agents.state import AgentWorkflowState


def build_agent_graph():
    """Construct and compile the cyclical multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentWorkflowState)

    # 1. Register Agents as Graph Nodes
    workflow.add_node("research_evidence", retrieve_and_evaluate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", generate_and_verify_answer_node)

    # 2. Define Entry Point
    workflow.set_entry_point("research_evidence")

    # 3. Conditional Branching from Research Evidence Agent
    workflow.add_conditional_edges(
        "research_evidence",
        route_evidence_check,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer"
        }
    )

    # 4. Web Search flows directly to Answer Synthesis
    workflow.add_edge("web_search", "generate_answer")

    # 5. Answer Verification terminates the graph
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# Compiled Singleton Graph Instance
agentic_rag_graph = build_agent_graph()
