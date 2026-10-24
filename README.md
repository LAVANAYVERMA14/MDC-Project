# EvalMind

**An Agentic AI Engine for Subjective Answer Evaluation and Exam Preparation**

EvalMind is a local browser-based MVP for marking short answers against a
reference answer and rubric. It reports rubric coverage, semantic similarity as
supporting evidence, and suggested revision topics. No paid API is required.

## Features

| Feature | Status |
|---|---|
| Student and examiner views | Student view hides answer keys and sample answers; examiner view includes technical details |
| Question, reference answer, maximum marks, rubric, and student answer | Sample questions or examiner-created custom questions |
| Semantic similarity | Sentence Transformer `all-MiniLM-L6-v2`; not treated as correctness |
| Criterion-wise marking | Baseline concept-coverage evaluator |
| Optional local LLM evaluator | Ollama structured JSON path; unit-tested with a fake model, not yet validated against a real model |
| Feedback | Rule-based score band, concepts, missing points, and suggestions |
| Student retry loop | Criterion checklist, topic nudges, word count, and per-question attempt history for the current session |
| Revision plan | Rubric topics and keyword-matched practice questions from the dataset |
| Sample dataset | Five CS questions and five answer types per question |
| Input validation | Non-empty text, positive marks, and rubric totals are validated |

## How marks are decided

1. The evaluator validates the question, reference answer, student answer, and
   rubric. Criterion marks must sum to the question maximum.
2. The baseline checks for rubric concepts using keyword matching and, when the
   embedding model is available, semantic matching against answer segments.
3. Marks are calculated and clamped in Python. Semantic similarity to the
   reference is displayed separately and does not change the score.
4. The baseline detects concept overlap, not factual correctness or negation.
   Its output is labelled **Concepts mentioned (not verified)**.

## Setup (Windows PowerShell)

Run commands from the project directory:

```powershell
cd "C:\Users\LOVISH\Desktop\pythhon\EvalMind"
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first evaluation downloads the sentence embedding model (about 90 MB).
Afterward it can run offline. This project has been run with Python 3.14.8.

## Run

```powershell
streamlit run app.py
```

The app opens in **Student** view, where sample questions are available without
showing the reference answer, rubric JSON, or sample answers. Use **Examiner**
view to inspect the answer key, rubric, sample answers, and technical scoring
details, or to create a custom question. Student attempts are kept for the
current browser session only; answer text is not added to attempt history.

For a custom question, enter one rubric criterion per line:

```text
Criterion name | marks | concept 1, concept 2
```

For example:

```text
Principle | 1 | LIFO
Operations | 1 | push, pop
```

Criterion marks must add up to the maximum marks. Custom questions have no
predefined revision topics or practice questions.

## Tests

```powershell
python -m pytest tests -v
```

Two tests in `tests/test_known_limitations.py` are marked `xfail`. They document
known baseline failures and should remain xfailed until a verifier is the
default and the desired behavior passes reliably.

## Optional local LLM (Ollama)

Install Ollama and pull a model such as `llama3.2`. Enable **Use Ollama LLM
fallback** in the app, or run:

```powershell
python evaluate_samples.py --llm
```

If Ollama is unavailable, evaluation falls back to the baseline and reports
the reason. Copy `.env.example` to `.env` to configure the URL or model. Do not
commit `.env`.

## Baseline evaluation on 25 sample answers

The report was generated with `python evaluate_samples.py` on 20 October 2026.
The preserved source data is [data/evaluation_report_baseline.csv](data/evaluation_report_baseline.csv).
Average similarity and marks by answer type (five answers per type):

| Answer type | Average similarity | Average marks |
|---|---:|---:|
| Correct | 0.91 | 98.0% |
| Incorrect (fluent but wrong) | 0.86 | 74.5% |
| Partial | 0.83 | 49.5% |
| Paraphrased | 0.60 | 40.5% |
| Irrelevant | 0.04 | 0.0% |

Marks were within the expected range for 11 of 25 answers: correct 5/5,
irrelevant 5/5, partial 1/5, paraphrased 0/5, and incorrect 0/5.

The baseline separates on-topic from irrelevant answers, but it cannot
reliably distinguish right from wrong when the answer reuses the reference
vocabulary. In these samples, incorrect answers had higher similarity than
partial answers and correct paraphrases for all five questions. Incorrect
answers scored above paraphrases on all five and above partial answers on four
(one tie). Correct paraphrases are also under-scored when their wording differs
from the rubric concepts.

These are 25 synthetic answers. The incorrect answers deliberately reuse
reference vocabulary with facts reversed, and expected ranges reflect the
project author's judgement. The results document a limitation; they are not a
claim of general performance. The semantic threshold has not been adjusted to
make this sample table look better.

## Limitations and next steps

- The baseline does not detect factual errors or negation; a fluent, wrong
  answer can earn marks.
- Semantic similarity measures closeness in meaning, not truth.
- The embedding model truncates long text (approximately 180 words).
- Practice questions are selected from a small predefined list by keyword
  overlap, not generated. Custom questions have no practice-question list.
- The optional local LLM path has not been evaluated against a real model and
  may also miss reversed claims.
- Results are not saved between sessions.
- Next: validate Ollama against the sample set, investigate a contradiction
  check (such as NLI), add retrieval and student history, and evaluate against
  human marks before considering deployment.

## Project structure

```text
EvalMind/
  app.py
  evaluate_samples.py
  similarity_report.py
  data/
    questions.json
    evaluation_report_baseline.csv
  src/
    evaluator.py
    semantic_analysis.py
    rubric_input.py
    feedback.py
    knowledge_gaps.py
  tests/
```
