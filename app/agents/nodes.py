import re
from datetime import datetime, timezone
from app.agents.answer_agent import get_answer_agent
from app.agents.evidence_checker import REALTIME_TRIGGERS, get_evidence_agent
from app.agents.state import AgentWorkflowState
from app.agents.web_search_agent import get_web_search_agent
from app.rag.hybrid_search import get_search_engine
from app.schemas.answer import AgentThoughtStep
from app.schemas.retrieval import RetrievalHit


def retrieve_and_evaluate_node(state: AgentWorkflowState) -> AgentWorkflowState:
    """Node 1: Retrieve hybrid candidates and evaluate document evidence directly."""
    question = state["question"]
    doc_id = state.get("document_id")
    top_k = state.get("top_k", 5)

    # 1. Hybrid Search (Milvus Dense + Persistent BM25 + Cross-Encoder Rerank)
    search_engine = get_search_engine()
    chunks = search_engine.search(
        query=question,
        top_k=top_k,
        filter_document_id=doc_id,
        mode="HYBRID_RERANK"
    )

    # Fallback: If doc_id was specified and chunks is empty, fetch all chunks for that document
    if not chunks and doc_id:
        try:
            from app.indexing.bm25_store import get_bm25_store
            doc_chunks = get_bm25_store().get_document_chunks(doc_id)
            if doc_chunks:
                chunks = [
                    RetrievalHit(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        filename=c.metadata.filename,
                        page_number=c.metadata.page_number,
                        chunk_index=c.metadata.chunk_index,
                        text=c.text,
                        score=1.0,
                        retrieval_source="DOC_FALLBACK"
                    )
                    for c in doc_chunks[:5]
                ]
        except Exception:
            pass

    state["retrieved_chunks"] = chunks
    answer_agent = get_answer_agent()

    # 2. Check for real-time / live knowledge triggers
    is_realtime = False
    for pat in REALTIME_TRIGGERS:
        if re.search(pat, question, re.IGNORECASE):
            is_realtime = True
            break

    if is_realtime:
        step = AgentThoughtStep(
            agent_name="ResearchEvidenceAgent",
            action="REALTIME_INTENT_DETECTED",
            rationale="Question requires live external information.",
            timestamp=datetime.now(timezone.utc)
        )
        state["evidence_is_sufficient"] = False
        state["web_search_needed"] = True
        state["thought_steps"] = state.get("thought_steps", []) + [step]
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        return state

    # 3. Check document inputs completely first!
    if chunks:
        doc_ans, doc_cites, conf = answer_agent.answer_from_documents(question, chunks)
        if doc_ans and "INSUFFICIENT_DOCUMENT_EVIDENCE" not in doc_ans.upper():
            # Sufficient answer found in documents
            step = AgentThoughtStep(
                agent_name="ResearchEvidenceAgent",
                action="DOCUMENT_ANSWER_SYNTHESIZED",
                rationale=f"Found conclusive evidence in ingested document(s) with {len(doc_cites)} citation(s).",
                timestamp=datetime.now(timezone.utc)
            )
            state["answer"] = doc_ans
            state["document_citations"] = doc_cites
            state["web_citations"] = []
            state["source_type"] = "DOCUMENTS"
            state["confidence_score"] = conf
            state["evidence_is_sufficient"] = True
            state["web_search_needed"] = False
            state["thought_steps"] = state.get("thought_steps", []) + [step]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return state

    # 4. Document context did NOT contain the answer
    step = AgentThoughtStep(
        agent_name="ResearchEvidenceAgent",
        action="INSUFFICIENT_DOC_EVIDENCE",
        rationale="Document context does not contain answer. Checking web search permission.",
        timestamp=datetime.now(timezone.utc)
    )
    state["evidence_is_sufficient"] = False
    state["web_search_needed"] = True
    state["thought_steps"] = state.get("thought_steps", []) + [step]
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state


def web_search_node(state: AgentWorkflowState) -> AgentWorkflowState:
    """Node 2: Fallback external web search when document evidence is insufficient."""
    question = state["question"]
    web_agent = get_web_search_agent()

    web_results = web_agent.search(question)

    step = AgentThoughtStep(
        agent_name="WebSearchAgent",
        action="WEB_SEARCH",
        rationale=f"Document evidence was insufficient. Retrieved {len(web_results)} live web passage(s).",
        timestamp=datetime.now(timezone.utc)
    )

    state["web_results"] = web_results
    state["thought_steps"] = state.get("thought_steps", []) + [step]
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    return state


def generate_and_verify_answer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    """Node 3: Grounded synthesis, citation extraction, and anti-hallucination verification."""
    # If answer was already generated from documents in Node 1, return immediately
    if state.get("answer") and state.get("evidence_is_sufficient"):
        return state

    question = state["question"]
    web_res = state.get("web_results", [])
    allow_web = state.get("allow_web_search", False)
    answer_agent = get_answer_agent()

    # 1. If web search was executed and returned results:
    if web_res:
        web_ans, web_cites, conf = answer_agent.answer_from_web(question, web_res)
        if web_ans and "INSUFFICIENT_WEB_EVIDENCE" not in web_ans.upper():
            step = AgentThoughtStep(
                agent_name="AnswerVerificationAgent",
                action="SYNTHESIZE_FROM_WEB",
                rationale=f"Synthesized response from live web search with {len(web_cites)} citation(s).",
                timestamp=datetime.now(timezone.utc)
            )
            state["answer"] = web_ans
            state["document_citations"] = []
            state["web_citations"] = web_cites
            state["confidence_score"] = conf
            state["source_type"] = "WEB"
            state["thought_steps"] = state.get("thought_steps", []) + [step]
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return state

    # 2. If documents did not have answer and web search was NOT enabled:
    if not allow_web:
        ans_text = (
            "The provided documents do not contain information to answer this question. "
            "Please enable the Web Search toggle if you would like to search external sources."
        )
    else:
        ans_text = (
            "Based on the provided documents and web search results, there is insufficient evidence to answer this question."
        )

    step = AgentThoughtStep(
        agent_name="AnswerVerificationAgent",
        action="NO_EVIDENCE_RESPONSE",
        rationale="Insufficient evidence across sources.",
        timestamp=datetime.now(timezone.utc)
    )
    state["answer"] = ans_text
    state["document_citations"] = []
    state["web_citations"] = []
    state["confidence_score"] = 0.0
    state["source_type"] = "INSUFFICIENT_EVIDENCE"
    state["thought_steps"] = state.get("thought_steps", []) + [step]
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state

