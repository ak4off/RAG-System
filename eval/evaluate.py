"""
Offline evaluation pipeline using RAGAS.

Usage:
    python -m eval.evaluate                        # run against Phase 1 pipeline
    python -m eval.evaluate --v2                   # run against Phase 2 pipeline
    python -m eval.evaluate --threshold 0.7        # fail if any metric drops below 0.7

Exit code 1 if any metric is below the threshold (for CI gate).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def run_eval(use_v2: bool = False, threshold: float = 0.7) -> dict:
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    with open(dataset_path) as f:
        items = json.load(f)

    if use_v2:
        from src.pipeline_v2 import RAGPipelineV2
        pipeline = RAGPipelineV2()
    else:
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running eval on {len(items)} questions...")
    for item in items:
        result = pipeline.query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([s["excerpt"] for s in result["sources"]])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores_dict = {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
    }

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores_dict.items():
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<25} {score:.3f}  [{status}]")
    print("=" * 50)

    failed = [k for k, v in scores_dict.items() if v < threshold]
    if failed:
        print(f"\nFAILED metrics (threshold={threshold}): {failed}")
        sys.exit(1)
    else:
        print(f"\nAll metrics above threshold ({threshold}). Build PASSED.")

    return scores_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    run_eval(use_v2=args.v2, threshold=args.threshold)
