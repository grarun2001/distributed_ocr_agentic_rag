"""Information Retrieval evaluation benchmarks computing Recall@K, MRR, and Precision@K."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set


@dataclass
class QueryTestCase:
    """A benchmark query paired with expected relevant chunk or document identifiers."""
    query: str
    expected_ids: Set[str]
    document_id: str = ""


@dataclass
class RetrievalMetrics:
    """Information retrieval evaluation scores for a specific retrieval mode and K."""
    mode: str
    k: int
    recall_at_k: float
    mrr: float
    precision_at_k: float
    total_queries: int


def calculate_metrics_for_mode(
    test_cases: List[QueryTestCase],
    search_func: Callable[[str, int, str], List[str]],
    mode: str,
    k: int
) -> RetrievalMetrics:
    """
    Evaluate retrieval quality across queries for a specific top_k and search mode.
    search_func takes (query, k, doc_filter) and returns list of retrieved chunk/doc IDs in ranked order.
    """
    if not test_cases:
        return RetrievalMetrics(mode=mode, k=k, recall_at_k=0.0, mrr=0.0, precision_at_k=0.0, total_queries=0)

    recall_hits = 0
    reciprocal_ranks = []
    precision_sum = 0.0

    for tc in test_cases:
        retrieved_ids = search_func(tc.query, k, tc.document_id)
        retrieved_top_k = retrieved_ids[:k]

        # 1. Recall@K (did at least one relevant document appear?)
        matched = any(rid in tc.expected_ids for rid in retrieved_top_k)
        if matched:
            recall_hits += 1

        # 2. MRR (reciprocal rank of first relevant match)
        first_rank = 0
        for idx, rid in enumerate(retrieved_top_k, start=1):
            if rid in tc.expected_ids:
                first_rank = idx
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)

        # 3. Precision@K (proportion of retrieved items that are relevant)
        relevant_in_k = sum(1 for rid in retrieved_top_k if rid in tc.expected_ids)
        precision_sum += (relevant_in_k / k) if k > 0 else 0.0

    total_q = len(test_cases)
    recall_at_k = round(recall_hits / total_q, 4)
    mrr = round(sum(reciprocal_ranks) / total_q, 4)
    precision_at_k = round(precision_sum / total_q, 4)

    return RetrievalMetrics(
        mode=mode,
        k=k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        precision_at_k=precision_at_k,
        total_queries=total_q
    )


def compare_retrieval_modes(
    test_cases: List[QueryTestCase],
    search_engine_dispatch: Callable[[str, int, str, str], List[str]],
    modes: List[str] = ["VECTOR_ONLY", "BM25_ONLY", "HYBRID_RRF", "HYBRID_RERANK"],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """
    Run multi-mode comparison benchmark quantifying the lift of hybrid RRF and Cross-Encoder reranking.
    search_engine_dispatch: func(query, top_k, mode, doc_id) -> list of IDs
    """
    benchmark_report: Dict[str, Any] = {
        "modes": {},
        "summary": []
    }

    for mode in modes:
        mode_results = {}
        for k in k_values:
            search_fn = lambda q, top_k, doc_id, m=mode: search_engine_dispatch(q, top_k, m, doc_id)
            metrics = calculate_metrics_for_mode(test_cases, search_fn, mode=mode, k=k)
            mode_results[f"K={k}"] = {
                "recall": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "precision": metrics.precision_at_k
            }
            benchmark_report["summary"].append(metrics)
        benchmark_report["modes"][mode] = mode_results

    return benchmark_report
