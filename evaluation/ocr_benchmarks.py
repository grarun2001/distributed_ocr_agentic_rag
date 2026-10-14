"""OCR Quantitative Benchmark evaluating Character Error Rate (CER) and Word Error Rate (WER)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    Compute Levenshtein edit distance (minimum insertions, deletions, substitutions)
    between two sequences using dynamic programming with O(min(M, N)) space.
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Keep track of previous and current row
    previous_row = list(range(len2 + 1))

    for i, elem1 in enumerate(seq1):
        current_row = [i + 1] * (len2 + 1)
        for j, elem2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if elem1 == elem2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[len2]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) = EditDistance(ref, hyp) / len(ref).
    Returns float in range [0.0, inf).
    """
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = levenshtein_distance(list(ref_clean), list(hyp_clean))
    return dist / len(ref_clean)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) = EditDistance(ref_words, hyp_words) / len(ref_words).
    Returns float in range [0.0, inf).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


@dataclass
class OCRBenchmarkResult:
    """Quantitative performance score comparing ground-truth text against OCR output."""
    ground_truth_chars: int
    predicted_chars: int
    ground_truth_words: int
    predicted_words: int
    cer: float
    wer: float
    char_accuracy: float
    word_accuracy: float
    passed: bool


def evaluate_ocr_text(
    ground_truth: str,
    ocr_predicted: str,
    max_cer: float = 0.05,
    max_wer: float = 0.10
) -> OCRBenchmarkResult:
    """
    Evaluate a single OCR extraction against reference ground truth.
    Defaults: CER <= 5% and WER <= 10% for production OCR acceptance.
    """
    cer = calculate_cer(ground_truth, ocr_predicted)
    wer = calculate_wer(ground_truth, ocr_predicted)
    char_acc = max(0.0, 1.0 - cer)
    word_acc = max(0.0, 1.0 - wer)
    passed = (cer <= max_cer) and (wer <= max_wer)

    return OCRBenchmarkResult(
        ground_truth_chars=len(ground_truth.strip()),
        predicted_chars=len(ocr_predicted.strip()),
        ground_truth_words=len(ground_truth.strip().split()),
        predicted_words=len(ocr_predicted.strip().split()),
        cer=round(cer, 4),
        wer=round(wer, 4),
        char_accuracy=round(char_acc, 4),
        word_accuracy=round(word_acc, 4),
        passed=passed
    )


def evaluate_ocr_dataset(
    samples: List[Tuple[str, str]],
    max_cer: float = 0.05,
    max_wer: float = 0.10
) -> Dict[str, Any]:
    """
    Evaluate a collection of (ground_truth, ocr_predicted) document pairs.
    Returns aggregate statistics and pass rate.
    """
    if not samples:
        return {
            "total_samples": 0,
            "mean_cer": 0.0,
            "mean_wer": 0.0,
            "mean_char_accuracy": 1.0,
            "mean_word_accuracy": 1.0,
            "pass_rate": 1.0,
            "results": []
        }

    results: List[OCRBenchmarkResult] = []
    for ref, hyp in samples:
        res = evaluate_ocr_text(ref, hyp, max_cer=max_cer, max_wer=max_wer)
        results.append(res)

    total = len(results)
    mean_cer = sum(r.cer for r in results) / total
    mean_wer = sum(r.wer for r in results) / total
    mean_char_acc = sum(r.char_accuracy for r in results) / total
    mean_word_acc = sum(r.word_accuracy for r in results) / total
    passed_count = sum(1 for r in results if r.passed)

    return {
        "total_samples": total,
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "mean_char_accuracy": round(mean_char_acc, 4),
        "mean_word_accuracy": round(mean_word_acc, 4),
        "passed_samples": passed_count,
        "pass_rate": round(passed_count / total, 4),
        "results": results
    }
