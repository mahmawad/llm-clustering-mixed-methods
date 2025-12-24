"""Utility helpers for classifying prompts into predefined categories."""

from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
import toml
import fasttext

model = fasttext.load_model("lid.176.ftz")

# Load API key from config
config_path = Path(__file__).parent / "config" / "config.toml"
config = toml.load(config_path)
api_key = config["api"]["openai_api_key"]

CATEGORY_METADATA: Dict[str, Dict[str, str]] = {
    "D.I": {
        "title": "Identification",
        "description": (
            "The user mainly DESCRIBES a problem, task, or situation and wants to DEFINE "
            "what the issue is. They do NOT primarily ask the chatbot to explain content, "
            "solve exercises, or provide new information."
        ),
    },
    "D.G": {
        "title": "Goal Setting and Requirement Clarification",
        "description": (
            "The user sets learning goals or clarifies what is expected in a course, exam, "
            "or task."
        ),
    },
    "S.S": {
        "title": "Search",
        "description": (
            "The user wants NEW INFORMATION, explanations, or definitions from the chatbot "
            "(e.g. \"Erkläre mir...\", \"Was ist...\", \"Gibt es Studien zu...\"). "
            "The focus is on obtaining knowledge, not on rewriting or solving given exercises."
        ),
    },
    "S.SL": {
        "title": "Select",
        "description": (
            "The user asks for summaries, key points, or extractions "
            "(e.g. \"Fasse zusammen\", \"Nenne mir die wichtigsten Punkte\")."
        ),
    },
    "S.EQ": {
        "title": "Evaluation of Information Quality",
        "description": (
            "The user asks about credibility, reliability, or quality of information or sources."
        ),
    },
    "E.RV": {
        "title": "Review",
        "description": (
            "The user wants existing work or answers to be checked, corrected, or improved "
            "(e.g. \"Überprüfe meine Lösung\", \"Prüfe meinen Text\")."
        ),
    },
    "E.O": {
        "title": "Organise",
        "description": (
            "The user wants help STRUCTURING or CATEGORIZING content "
            "(e.g. Gliederungen, Mindmaps, Sortieren in Kategorien), not explanations."
        ),
    },
    "E.RF": {
        "title": "Reformatting and Reworking",
        "description": (
            "The user wants to TRANSFORM given content (e.g. Übersetzen, Umformulieren, "
            "in Stichworte/Diagramme umwandeln) OR asks the chatbot to solve exercises "
            "STEP BY STEP based on provided material."
        ),
    },
    "E.RH": {
        "title": "Rehearse",
        "description": (
            "The user wants to PRACTISE, e.g. Quizfragen, Übungsaufgaben, Wiederholungsfragen "
            "zu gelerntem Stoff."
        ),
    },
    "R.ET": {
        "title": "Task Evaluation",
        "description": (
            "The user explicitly assesses or critiques the quality, correctness, or completeness "
            "of a solution, assignment, or work product (e.g. \"Ist diese Lösung richtig?\", "
            "\"Habe ich die Aufgabe korrekt bearbeitet?\", \"Ist diese Antwort ausreichend?\"). "
            "The user is asking for JUDGMENT or FEEDBACK on their own or another's work quality, "
            "NOT asking how to improve or fix it."
        ),
    },
    "R.ES": {
        "title": "Self Evaluation",
        "description": (
            "The user reflects on their OWN UNDERSTANDING, LEARNING GAPS, or READINESS. They "
            "question what they understand or don't understand (e.g. \"Ich verstehe das nicht\", "
            "\"Was bedeutet das genau?\", \"Wo liegt mein Verständnislücke?\"). This is "
            "META-COGNITIVE reflection, not a request to explain the concept itself."
        ),
    },
    "OTHER": {
        "title": "Other",
        "description": (
            "Only if the query clearly does NOT fit any of the above categories "
            "(e.g. off-topic, meta-questions about the chatbot itself, greetings without substance)."
        ),
    },
}

CATEGORY_GROUPS = [
    ("Defining", ["D.I", "D.G"]),
    ("Seeking", ["S.S", "S.SL", "S.EQ"]),
    ("Engaging", ["E.RV", "E.O", "E.RF", "E.RH"]),
    ("Reflecting", ["R.ET", "R.ES"]),
    ("Other", ["OTHER"]),
]

CATEGORY_ORDER = [code for _, codes in CATEGORY_GROUPS for code in codes]

CATEGORY_CODE_ALIASES: Dict[str, str] = {}
for _code in CATEGORY_ORDER:
    canonical = _code.upper()
    CATEGORY_CODE_ALIASES[canonical] = _code
    stripped = canonical.replace(".", "")
    CATEGORY_CODE_ALIASES[stripped] = _code

_selected_category_codes: Optional[List[str]] = None

classification_prompt_template = """
You are an AI model specialized in analyzing and assigning user queries
according to a predefined categorization scheme.

Your task is to examine the given inputs and assign them a suitable category
from the following list:

{category_section}

You are given a single topic consisting of multiple user queries.

Representative user queries:
{documents}

Decision rules:
1. If the user explicitly asks for an EXPLANATION, DEFINITION, or INFORMATION
   (e.g. "Erkläre", "Was ist", "Gibt es Studien"), prefer S.S over D.I.
2. If the user asks to SOLVE EXERCISES or work through tasks based on slides,
   examples, or given text, prefer E.RF.
3. If the user wants EXISTING text or solutions to be CHECKED or CORRECTED,
   prefer E.RV.
4. Use D.I only if the main action is to DESCRIBE or DEFINE a problem or task,
   without primarily asking for explanations, solutions, or practice.
5. Use E.O only if the main request is to STRUCTURE or ORGANIZE content
   (outline, categories, ordering), not to explain or solve.
6. CRITICAL - Distinguish R.ES from S.S:
   - R.ES: "Ich verstehe X nicht" or "Was verstehe ich hier falsch?" (self-reflection about understanding)
   - S.S: "Erkläre mir X" or "Was ist X?" (requesting new information/explanation)
   - R.ES is meta-cognitive (about the user's own understanding gaps),
     S.S is informational (requesting knowledge from the chatbot).
7. CRITICAL - Distinguish R.ET from E.RV:
   - R.ET: "Ist meine Lösung richtig?" or "Ist das ausreichend?" (asking for judgment/assessment)
   - E.RV: "Überprüfe meine Lösung und korrigiere Fehler" (asking for checking AND improvement)
   - R.ET is evaluation/assessment, E.RV is review/correction.
8. Use OTHER sparingly - most queries fit into one of the main categories.
   Only use OTHER if the query is off-topic or meta (e.g., asking about the chatbot system itself).

Task:
1. Analyze the user queries and their shared intent.
2. Assign the most appropriate category to this topic from:
   D.I, D.G, S.S, S.SL, S.EQ, E.RV, E.O, E.RF, E.RH, R.ET, R.ES, OTHER.

Important:
- Output ONLY the category code (exactly one of:
  D.I, D.G, S.S, S.SL, S.EQ, E.RV, E.O, E.RF, E.RH, R.ET, R.ES, OTHER).
- Do not include any explanation or additional text.
"""


def _short_description(text: str) -> str:
    """Return the first sentence so the menu stays readable."""
    if not text:
        return ""
    period_index = text.find(".")
    if period_index == -1:
        return text
    return text[: period_index + 1]


def prompt_user_for_category_selection() -> List[str]:
    """Ask the user which categories should appear in the prompt."""
    print("\nChoose the topics that should be included in the classification prompt:")
    for idx, code in enumerate(CATEGORY_ORDER, start=1):
        info = CATEGORY_METADATA[code]
        summary = _short_description(info["description"])
        print(f"  {idx:>2}. {info['title']} ({code})")
        if summary:
            print(f"       {summary}")

    try:
        selection = input(
            "Enter numbers, category codes (e.g. D.I), or 'all' for every topic: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo selection made; defaulting to all categories.")
        return CATEGORY_ORDER.copy()

    if not selection or selection.lower() in {"all", "*"}:
        return CATEGORY_ORDER.copy()

    tokens = selection.replace(",", " ").split()
    chosen: List[str] = []
    seen = set()
    for token in tokens:
        code = None
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(CATEGORY_ORDER):
                code = CATEGORY_ORDER[index]
        else:
            normalized = token.upper().replace(" ", "")
            code = CATEGORY_CODE_ALIASES.get(normalized)
        if code and code not in seen:
            chosen.append(code)
            seen.add(code)

    if not chosen:
        print("No valid selection detected; defaulting to all categories.")
        return CATEGORY_ORDER.copy()

    return chosen


def get_active_category_codes() -> List[str]:
    """Return the cached selection or prompt the user once."""
    global _selected_category_codes
    if _selected_category_codes is None:
        _selected_category_codes = prompt_user_for_category_selection()
        selection_summary = ", ".join(
            f"{CATEGORY_METADATA[code]['title']} ({code})" for code in _selected_category_codes
        )
        print(f"Using categories: {selection_summary}")
    return _selected_category_codes


def build_category_section(selected_codes: List[str]) -> str:
    """Render only the selected categories for the prompt."""
    lines: List[str] = ["# Categories"]
    for group_label, group_codes in CATEGORY_GROUPS:
        group_selected = [code for code in group_codes if code in selected_codes]
        if not group_selected:
            continue
        lines.append(f"## {group_label}")
        for code in group_selected:
            info = CATEGORY_METADATA.get(code)
            if not info:
                continue
            lines.append(f"- {info['title']} ({code}): {info['description']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def get_selected_category_codes() -> List[str]:
    """Return the cached list of selected category codes."""
    # Copy to prevent callers from mutating the shared cache.
    return get_active_category_codes().copy()


# Create the OpenAI API client
client = OpenAI(api_key=api_key)


def classify_prompt(doc: Optional[str], model: str = "gpt-4o-mini") -> str:
    """Classify a single prompt and return the action code."""
    text = (doc or "").strip()
    if not text:
        return "OTHER"

    categories = get_active_category_codes()
    category_section = build_category_section(categories)
    documents_section = f"- {text}"
    filled_prompt = classification_prompt_template.format(
        category_section=category_section,
        documents=documents_section,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": filled_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Error classifying prompt: {exc}")
        return "ERROR"


def detect_language(text):
    """Detect language using the FastText model."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        pred = model.predict(text.replace('\n', ' '))
        lang = pred[0][0].replace('__label__', '')
        confidence = pred[1][0]
        return lang
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "error"
