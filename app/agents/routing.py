"""Conditional routing edge logic for LangGraph multi-agent state graph."""

from app.agents.state import AgentWorkflowState
from app.core.logging import get_logger

logger = get_logger("agent_routing")


def route_evidence_check(state: AgentWorkflowState) -> str:
    """Decide whether to branch to WebSearchAgent or directly to AnswerVerificationAgent."""
    # If already answered from documents, proceed directly to answer completion
    if state.get("evidence_is_sufficient", False) and state.get("answer"):
        return "generate_answer"

    need_web = state.get("web_search_needed", False)
    allow_web = state.get("allow_web_search", False)
    iterations = state.get("iteration_count", 0)

    # Bound recursion limit guardrail (prevent infinite loops)
    if iterations >= 5:
        logger.warning("Agent workflow reached maximum iteration bound (5). Forcing answer synthesis.")
        return "generate_answer"

    if need_web and allow_web:
        logger.info("Routing to 'web_search' node (document lacked evidence and web search is enabled).")
        return "web_search"
    else:
        logger.info("Routing to 'generate_answer' node (document evidence sufficient or web search disabled).")
        return "generate_answer"

