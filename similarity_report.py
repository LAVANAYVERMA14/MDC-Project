"""Runs semantic similarity on every sample answer in data/questions.json
and writes data/similarity_report.csv.

Usage (from the EvalMind/ folder):  python similarity_report.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.semantic_analysis import ModelLoadError, SemanticAnalyzer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "questions.json"
CSV_PATH = ROOT / "data" / "similarity_report.csv"
LABEL_ORDER = ["correct", "paraphrased", "partial", "incorrect", "irrelevant"]


def main() -> int:
    with open(DATA_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    analyzer = SemanticAnalyzer()
    rows = []
    print("Loading model (first run downloads ~90 MB)...")

    try:
        for q in questions:
            for ans in q["sample_answers"]:
                result = analyzer.compare(ans["text"], q["reference_answer"])
                rows.append({
                    "question_id": q["id"],
                    "topic": q["topic"],
                    "label": ans["label"],
                    "similarity": round(result.score, 3),
                    "band": result.label,
                    "expected_marks": f"{ans['expected_score_range'][0]}-{ans['expected_score_range'][1]}",
                    "max_marks": q["max_marks"],
                })
    except ModelLoadError as exc:
        print(f"\nERROR: {exc}")
        return 1

    # ---- Per-question table ----
    for q in questions:
        print(f"\n{q['id']} | {q['topic']} | {q['question']}")
        print(f"  {'label':<12}{'similarity':<12}{'band':<10}{'expected marks (of ' + str(q['max_marks']) + ')'}")
        q_rows = [r for r in rows if r["question_id"] == q["id"]]
        q_rows.sort(key=lambda r: LABEL_ORDER.index(r["label"]))
        for r in q_rows:
            print(f"  {r['label']:<12}{r['similarity']:<12}{r['band']:<10}{r['expected_marks']}")

    # ---- Average similarity by label ----
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["similarity"])
    print("\nAverage similarity by answer type (across all questions):")
    for label in LABEL_ORDER:
        vals = by_label[label]
        print(f"  {label:<12}{sum(vals) / len(vals):.3f}")

    # ---- Observations computed from the data ----
    print("\nObservations computed from this run:")
    incorrect_vs_partial = 0
    paraphrased_below_incorrect = 0
    for q in questions:
        sims = {r["label"]: r["similarity"] for r in rows if r["question_id"] == q["id"]}
        if sims["incorrect"] >= sims["partial"]:
            incorrect_vs_partial += 1
        if sims["paraphrased"] <= sims["incorrect"]:
            paraphrased_below_incorrect += 1
    n = len(questions)
    print(f"  Wrong-but-fluent answer scored >= the partially correct answer in "
          f"{incorrect_vs_partial}/{n} questions.")
    print(f"  Correct paraphrase scored <= the wrong-but-fluent answer in "
          f"{paraphrased_below_incorrect}/{n} questions.")

    # ---- CSV ----
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
