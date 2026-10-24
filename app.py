"""Student and examiner interface for EvalMind."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.evaluator import EvaluationResult, InvalidInputError, evaluate_answer
from src.feedback import build_feedback
from src.knowledge_gaps import find_knowledge_gaps, suggest_practice_questions
from src.rubric_input import parse_rubric_text, rubric_to_text
from src.semantic_analysis import SemanticAnalysisError, SemanticAnalyzer

DATA_PATH = Path(__file__).resolve().parent / "data" / "questions.json"
MAX_MARKS_UI = 100.0
RUBRIC_HELP = "One criterion per line: Criterion name | marks | concept 1, concept 2"
HISTORY_LIMIT = 200
EPSILON = 1e-9


@st.cache_data
def load_questions() -> List[Dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["questions"]


@st.cache_resource
def get_semantic_analyzer() -> SemanticAnalyzer:
    return SemanticAnalyzer()


def make_signature(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def status_for_marks(awarded: float, maximum: float) -> tuple[str, str]:
    if awarded >= maximum - EPSILON:
        return "check_circle", "Full"
    if awarded > EPSILON:
        return "pending", "Partial"
    return "radio_button_unchecked", "Not yet"


def render_student_checklist(result: EvaluationResult) -> None:
    st.markdown("#### Rubric checklist")
    for criterion in result.criteria:
        icon, label = status_for_marks(
            criterion.marks_awarded,
            criterion.max_marks,
        )
        with st.container(border=True):
            st.markdown(
                f":material/{icon}: **{label}** · **{criterion.name}**  \n"
                f"{criterion.marks_awarded:g} / {criterion.max_marks:g} marks"
            )
            if label != "Full":
                topic = criterion.revision_topic or criterion.name
                with st.expander(f"Need a nudge? Think about {topic}"):
                    st.write(
                        "Try explaining this idea in your own words, then edit "
                        "your answer and check again."
                    )


def render_attempt_history(context_signature: str) -> None:
    attempts = [
        attempt
        for attempt in st.session_state["attempt_history"]
        if attempt["context"] == context_signature
    ]
    if not attempts:
        return

    st.markdown("#### Your attempts")
    st.caption(
        "Your attempt history stays in this browser session and records scores, "
        "not your answer text."
    )
    for index, attempt in enumerate(attempts, start=1):
        change = ""
        if index > 1:
            delta = attempt["score"] - attempts[index - 2]["score"]
            change = f" ({delta:+g} from the previous attempt)"
        st.write(
            f"**Attempt {index}** · {attempt['score']:g}/{attempt['maximum']:g}"
            f"{change} · {attempt['time']}"
        )

    if len(attempts) > 1:
        st.line_chart(
            {"Marks": [attempt["score"] for attempt in attempts]},
            x_label="Attempt",
            y_label="Marks",
        )


def render_student_results(result: EvaluationResult) -> None:
    st.subheader("Your results")
    fraction = result.total_marks / result.max_marks if result.max_marks else 0.0
    st.metric(
        "Concept coverage marks",
        f"{result.total_marks:g} / {result.max_marks:g}",
    )
    st.progress(fraction, text=f"{fraction:.0%} of rubric marks covered")
    st.info(
        "Concept coverage shows whether rubric ideas appear in your answer. "
        "It does not verify that every statement is factually correct."
    )
    render_student_checklist(result)

    gaps = [
        criterion
        for criterion in result.criteria
        if criterion.marks_awarded < criterion.max_marks - EPSILON
    ]
    if gaps:
        st.markdown("#### Try this next")
        st.write(
            "Revise the items marked **Partial** or **Not yet**, then submit "
            "your updated answer to compare attempts."
        )
        topics = list(
            dict.fromkeys(
                criterion.revision_topic or criterion.name for criterion in gaps
            )
        )
        st.caption("Consider reviewing: " + " · ".join(topics))
    else:
        st.success(
            "Every rubric criterion received coverage marks. Review your answer "
            "once more for accuracy before you finish."
        )


def render_examiner_results(
    result: EvaluationResult,
    reference: str,
    rubric: List[Dict[str, Any]],
    student_answer: str,
) -> None:
    st.subheader("Examiner panel")
    with st.expander("Answer key and rubric", expanded=False):
        st.markdown("**Reference answer**")
        st.write(reference)
        st.markdown("**Rubric data**")
        st.code(
            json.dumps(rubric, indent=2, ensure_ascii=False),
            language="json",
        )

    with st.expander("Technical scoring details", expanded=False):
        st.markdown(f"**Evaluator:** `{result.method}`")
        if result.warnings:
            st.warning("\n\n".join(result.warnings))
        if result.fallback_reason:
            st.info(f"Fallback reason: {result.fallback_reason}")
        try:
            similarity = get_semantic_analyzer().compare(student_answer, reference)
        except SemanticAnalysisError as exc:
            st.warning(f"Semantic similarity could not be computed: {exc}")
        else:
            st.markdown("**Semantic similarity (supporting evidence only)**")
            st.write(
                f"{similarity.display_score:.2f} · {similarity.label.capitalize()}"
            )
            st.caption(similarity.interpretation)
            if similarity.truncation_warning:
                st.warning(similarity.truncation_warning)
            st.caption(similarity.disclaimer)

    feedback = build_feedback(result)
    st.markdown("**Detailed feedback**")
    st.write(feedback.summary)
    st.markdown(f"**{feedback.strengths_title}**")
    if feedback.strengths:
        st.markdown("\n".join(f"- {item}" for item in feedback.strengths))
    else:
        st.write("No rubric points were earned.")

    st.markdown("**Missing points and suggestions**")
    if feedback.missing:
        st.markdown("\n".join(f"- {item}" for item in feedback.missing))
    else:
        st.write("No rubric points are missing.")
    if feedback.suggestions:
        st.markdown("\n".join(f"- {item}" for item in feedback.suggestions))

    st.markdown("**Criterion-level evidence**")
    for criterion in result.criteria:
        with st.container(border=True):
            st.markdown(
                f"**{criterion.name}** · "
                f"{criterion.marks_awarded:g}/{criterion.max_marks:g}"
            )
            st.write(criterion.explanation)
            if criterion.matched_concepts:
                st.caption("Matched: " + ", ".join(criterion.matched_concepts))
            if criterion.missing_concepts:
                st.caption("Missing: " + ", ".join(criterion.missing_concepts))
            if criterion.factual_errors:
                st.warning("Reported factual errors: " + ", ".join(criterion.factual_errors))


def render_examiner_revision(
    result: EvaluationResult,
    question: Optional[Dict[str, Any]],
) -> None:
    st.markdown("**Revision plan**")
    gaps = find_knowledge_gaps(result, question)
    if not gaps:
        st.success("No gaps found against the rubric for this answer.")
        return

    st.dataframe(
        [
            {
                "Revision topic": gap.revision_topic,
                "Criterion": gap.criterion,
                "Marks lost": gap.marks_lost,
                "Concepts to revisit": ", ".join(gap.missing_concepts),
            }
            for gap in gaps
        ],
        hide_index=True,
    )
    practice = suggest_practice_questions(gaps, question)
    if practice:
        st.markdown("**Suggested practice questions**")
        st.markdown(
            "\n".join(
                f"- {item.question}  \n  *{item.reason}*" for item in practice
            )
        )
    else:
        st.caption("No predefined practice questions are available for this question.")


st.set_page_config(page_title="EvalMind", page_icon="🧠", layout="wide")
st.session_state.setdefault("attempt_history", [])
st.session_state.setdefault("last_evaluation", None)
st.session_state.setdefault("last_evaluation_signature", None)
st.session_state.setdefault("last_evaluated_answer", None)

st.title("EvalMind")
st.caption(
    "Practice an answer, see which rubric areas are covered, and try again."
)
audience = st.segmented_control(
    "View",
    ["Student", "Examiner"],
    default="Student",
    key="audience_view",
)
is_examiner = audience == "Examiner"

if is_examiner:
    source = st.sidebar.segmented_control(
        "Question source",
        ["Sample question", "Custom question"],
        default="Sample question",
        key="examiner_question_source",
    )
else:
    source = "Sample question"

selected_question: Optional[Dict[str, Any]] = None
rubric: Optional[List[Dict[str, Any]]] = None
rubric_error = ""
sample_answers: Dict[str, str] = {}
sample_label = "Write my own"

if source == "Sample question":
    questions = load_questions()
    question_ids = [question["id"] for question in questions]
    if is_examiner:
        selected_id = st.sidebar.selectbox("Question", question_ids, index=0)
    else:
        question_by_id = {question["id"]: question for question in questions}
        selected_id = st.selectbox(
            "Choose a question",
            question_ids,
            index=0,
            format_func=lambda question_id: question_by_id[question_id]["topic"],
        )
    selected_question = next(
        question for question in questions if question["id"] == selected_id
    )
    question_text = selected_question["question"]
    reference = selected_question["reference_answer"]
    max_marks = selected_question["max_marks"]
    rubric = selected_question["rubric"]

    if is_examiner:
        manual_rubric = st.sidebar.toggle(
            "Edit rubric text manually",
            value=False,
            help="The original dataset rubric is used unless you change the text.",
        )
        if manual_rubric:
            default_rubric_text = rubric_to_text(selected_question["rubric"])
            rubric_text = st.sidebar.text_area(
                "Rubric text",
                value=default_rubric_text,
                height=220,
                help=RUBRIC_HELP,
            )
            if rubric_text.strip() == default_rubric_text.strip():
                rubric = selected_question["rubric"]
            else:
                try:
                    rubric = parse_rubric_text(rubric_text)
                    st.sidebar.caption(
                        "Edited rubric in use. Dataset min_concepts are not applied."
                    )
                except InvalidInputError as exc:
                    st.sidebar.error(str(exc))
                    rubric = None
                    rubric_error = str(exc)

    st.subheader(selected_question["topic"])
    st.markdown(f"**Question**  \n{question_text}")
    sample_answers = {
        item["label"]: item["text"] for item in selected_question["sample_answers"]
    }
    if is_examiner:
        with st.expander("Answer key and rubric"):
            st.markdown("**Reference answer**")
            st.write(reference)
            st.markdown("**Rubric data**")
            st.code(
                json.dumps(rubric or selected_question["rubric"], indent=2, ensure_ascii=False),
                language="json",
            )
        sample_label = st.selectbox(
            "Load a sample answer",
            ["Write my own"] + sorted(sample_answers),
            help="Sample answers are for examiner demonstrations and testing.",
            key=f"sample_label_{selected_id}",
            persist_state="session",
        )
else:
    st.subheader("Create a custom question")
    st.caption(
        "Enter the question, expected answer, and a rubric whose criterion "
        "marks add up to the maximum."
    )
    question_text = st.text_area(
        "Question",
        key="custom_question",
        height=80,
        placeholder="Enter the question you want to evaluate.",
    )
    reference = st.text_area(
        "Reference answer",
        key="custom_reference",
        height=140,
        placeholder="Enter the key facts or expected answer.",
    )
    max_marks = st.number_input(
        "Maximum marks",
        min_value=0.0,
        max_value=MAX_MARKS_UI,
        value=5.0,
        step=0.5,
        key="custom_max_marks",
        help="The rubric criterion marks must add up to this value.",
    )
    custom_rubric_text = st.text_area(
        "Marking rubric",
        key="custom_rubric",
        height=130,
        help=RUBRIC_HELP,
        placeholder="Principle | 1 | LIFO\nOperations | 1 | push, pop",
    )
    st.caption(
        "One criterion per line: name | marks | concepts. "
        "Criterion marks must add up to the maximum marks."
    )
    if not custom_rubric_text.strip():
        rubric_error = "Enter a marking rubric."
    else:
        try:
            rubric = parse_rubric_text(custom_rubric_text)
        except InvalidInputError as exc:
            rubric_error = str(exc)

if source == "Sample question":
    answer_key = (
        f"examiner_answer_{selected_id}_{sample_label}"
        if is_examiner
        else f"student_answer_{selected_id}"
    )
    answer_default = (
        sample_answers[sample_label]
        if is_examiner and sample_label in sample_answers
        else ""
    )
else:
    answer_key = "examiner_custom_answer"
    answer_default = ""

st.markdown("#### Your answer")
student_answer = st.text_area(
    "Student answer",
    key=answer_key,
    value=answer_default,
    height=180,
    placeholder="Write your answer in your own words.",
    help="Your draft stays here when you edit and check it again.",
    persist_state="session",
)
word_count = len(student_answer.split())
st.caption(f"{word_count} {'word' if word_count == 1 else 'words'}")

use_llm = False
if is_examiner:
    use_llm = st.sidebar.toggle(
        "Use Ollama when available",
        value=False,
        help="If Ollama is unavailable, evaluation falls back to the baseline.",
    )

context_signature = make_signature(
    {
        "source": source,
        "question_id": selected_question["id"] if selected_question else None,
        "question": question_text,
        "reference": reference,
        "maximum": max_marks,
        "rubric": rubric,
    }
)
evaluation_signature = make_signature(
    {
        "context": context_signature,
        "use_llm": use_llm,
    }
)
context_attempts = [
    attempt
    for attempt in st.session_state["attempt_history"]
    if attempt["context"] == context_signature
]
if is_examiner:
    check_button_label = "Evaluate answer"
elif context_attempts:
    check_button_label = "Edit and re-check"
else:
    check_button_label = "Check my answer"

if st.button(
    check_button_label,
    type="primary",
    icon=":material/auto_awesome:",
    width="stretch",
):
    if rubric is None:
        st.error(f"Fix the rubric before evaluating: {rubric_error or 'Check rubric format.'}")
    elif not student_answer.strip():
        st.warning("Write an answer before you check it.")
    else:
        analyzer = get_semantic_analyzer()
        try:
            with st.status("Reading your answer...", expanded=False) as progress:
                progress.update(label="Checking ideas against the rubric...")
                result = evaluate_answer(
                    question_text,
                    reference,
                    max_marks,
                    rubric,
                    student_answer,
                    use_llm=use_llm,
                    analyzer=analyzer,
                )
                progress.update(
                    label="Your feedback is ready.",
                    state="complete",
                )
        except InvalidInputError as exc:
            st.error(f"Please check your inputs: {exc}")
        except Exception as exc:
            st.error(f"Evaluation could not be completed: {exc}")
        else:
            st.session_state["last_evaluation"] = result
            st.session_state["last_evaluation_signature"] = evaluation_signature
            st.session_state["last_evaluated_answer"] = student_answer
            st.session_state["attempt_history"].append(
                {
                    "context": context_signature,
                    "score": result.total_marks,
                    "maximum": result.max_marks,
                    "time": datetime.now().strftime("%H:%M"),
                    "method": result.method,
                }
            )
            st.session_state["attempt_history"] = st.session_state[
                "attempt_history"
            ][-HISTORY_LIMIT:]
            st.rerun()

attempts = [
    attempt
    for attempt in st.session_state["attempt_history"]
    if attempt["context"] == context_signature
]
if attempts:
    render_attempt_history(context_signature)

result = st.session_state.get("last_evaluation")
if (
    result is not None
    and st.session_state.get("last_evaluation_signature") == evaluation_signature
):
    st.divider()
    evaluated_answer = st.session_state.get("last_evaluated_answer") or ""
    if evaluated_answer != student_answer:
        st.caption(
            "Showing feedback for your previous answer. Check again to update it."
        )
    if is_examiner:
        st.subheader("Evaluation result")
        score_columns = st.columns(2)
        score_columns[0].metric(
            "Marks",
            f"{result.total_marks:g} / {result.max_marks:g}",
        )
        score_columns[1].metric("Evaluator method", result.method)
        render_examiner_results(result, reference, rubric or [], evaluated_answer)
        render_examiner_revision(
            result,
            selected_question if source == "Sample question" else None,
        )
    else:
        render_student_results(result)
        st.divider()
        st.markdown("**Was this feedback helpful?**")
        feedback_rating = st.feedback(
            "thumbs",
            key=f"feedback_rating_{evaluation_signature}",
        )
        if feedback_rating is not None:
            st.success("Thanks for letting us know.")
