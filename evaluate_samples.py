"""Evaluates every sample answer and compares with its expected score range.

Usage:
    python evaluate_samples.py          # baseline evaluator
    python evaluate_samples.py --llm    # try Ollama (falls back to baseline)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.evaluator import evaluate_answer
from src.semantic_analysis import SemanticAnalysisError, SemanticAnalyzer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "questions.json"
CSV_PATH = ROOT / "data" / "evaluation_report.csv"
LABEL_ORDER = ["correct", "paraphrased", "partial", "incorrect", "irrelevant"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use the Ollama LLM evaluator")
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    analyzer = SemanticAnalyzer()
    rows, methods = [], set()

    for q in questions:
        for ans in q["sample_answers"]:
            result = evaluate_answer(
                q["question"],
                q["reference_answer"],
                q["max_marks"],
                q["rubric"],
                ans["text"],
                use_llm=args.llm,
                analyzer=analyzer,
            )
            methods.add(result.method)
            try:
                sim = round(analyzer.compare(ans["text"], q["reference_answer"]).score, 3)
            except SemanticAnalysisError:
                sim = None
            low, high = ans["expected_score_range"]
            rows.append({
                "question_id": q["id"],
                "label": ans["label"],
                "similarity": sim,
                "marks": result.total_marks,
                "max_marks": q["max_marks"],
                "expected": f"{low}-{high}",
                "in_range": low <= result.total_marks <= high,
                "method": result.method,
            })

    print(f"Evaluator method(s) used: {', '.join(sorted(methods))}")
    for q in questions:
        print(f"\n{q['id']} | {q['question']}")
        print(f"  {'label':<12}{'similarity':<12}{'marks':<12}{'expected':<12}ok?")
        q_rows = sorted((r for r in rows if r["question_id"] == q["id"]), key=lambda r: LABEL_ORDER.index(r["label"]))
        for r in q_rows:
            sim = "n/a" if r["similarity"] is None else r["similarity"]
            print(
                f"  {r['label']:<12}{str(sim):<12}"
                f"{str(r['marks']) + '/' + str(q['max_marks']):<12}"
                f"{r['expected']:<12}{'yes' if r['in_range'] else 'NO'}"
            )

    by_label = defaultdict(lambda: [0, 0])
    for r in rows:
        by_label[r["label"]][1] += 1
        by_label[r["label"]][0] += r["in_range"]
    total_ok = sum(v[0] for v in by_label.values())
    print(f"\nMarks within the expected range: {total_ok}/{len(rows)}")
    for label in LABEL_ORDER:
        ok, n = by_label[label]
        print(f"  {label:<12}{ok}/{n}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
