"""Utility helpers for classifying prompts into predefined categories."""

from pathlib import Path
from typing import Optional
from openai import OpenAI
import toml
import fasttext 
model = fasttext.load_model("lid.176.ftz")

# Load API key from config
config_path = Path(__file__).parent / "config" / "config.toml"
config = toml.load(config_path)
api_key = config["api"]["openai_api_key"]

classification_prompt = """
You are an AI model specialized in analyzing and assigning user queries
according to a predefined categorization scheme.

Your task is to examine the given inputs and assign them a suitable category
from the following list:

# Categories
## Defining
- Identification (D.I): The user mainly DESCRIBES a problem, task, or situation
  and wants to DEFINE what the issue is. They do NOT primarily ask the chatbot
  to explain content, solve exercises, or provide new information.
- Goal Setting and Requirement Clarification (D.G): The user sets learning goals
  or clarifies what is expected in a course, exam, or task.

## Seeking
- Search (S.S): The user wants NEW INFORMATION, explanations, or definitions
  from the chatbot (e.g. "Erkläre mir...", "Was ist...", "Gibt es Studien zu...").
  The focus is on obtaining knowledge, not on rewriting or solving given exercises.
- Select (S.SL): The user asks for summaries, key points, or extractions
  (e.g. "Fasse zusammen", "Nenne mir die wichtigsten Punkte").
- Evaluation of Information Quality (S.EQ): The user asks about credibility,
  reliability, or quality of information or sources.

## Engaging
- Review (E.RV): The user wants existing work or answers to be checked,
  corrected, or improved (e.g. "Überprüfe meine Lösung", "Prüfe meinen Text").
- Organise (E.O): The user wants help STRUCTURING or CATEGORIZING content
  (e.g. Gliederungen, Mindmaps, Sortieren in Kategorien), not explanations.
- Reformatting and Reworking (E.RF): The user wants to TRANSFORM given content
  (e.g. Übersetzen, Umformulieren, in Stichworte/Diagramme umwandeln) OR
  asks the chatbot to solve exercises STEP BY STEP based on provided material.
- Rehearse (E.RH): The user wants to PRACTISE, e.g. Quizfragen, Übungsaufgaben,
  Wiederholungsfragen zu gelerntem Stoff.

## Reflecting
- Task Evaluation (R.ET): The user explicitly assesses or critiques the quality,
  correctness, or completeness of a solution, assignment, or work product
  (e.g. "Ist diese Lösung richtig?", "Habe ich die Aufgabe korrekt bearbeitet?",
  "Ist diese Antwort ausreichend?"). The user is asking for JUDGMENT or FEEDBACK
  on their own or another's work quality, NOT asking how to improve or fix it.
- Self Evaluation (R.ES): The user reflects on their OWN UNDERSTANDING, LEARNING GAPS,
  or READINESS. They question what they understand or don't understand
  (e.g. "Ich verstehe das nicht", "Was bedeutet das genau?", "Wo liegt mein Verständnislücke?").
  This is META-COGNITIVE reflection, not a request to explain the concept itself.

## Other
- OTHER: Only if the query clearly does NOT fit any of the above categories
  (e.g. off-topic, meta-questions about the chatbot itself, greetings without substance).

You are given a single topic consisting of multiple user queries.

Representative user queries:
[DOCUMENTS]

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

# Create the OpenAI API client
client = OpenAI(api_key=api_key)


def classify_prompt(doc: Optional[str], model: str = "gpt-4o-mini") -> str:
    """Classify a single prompt and return the action code."""
    text = (doc or "").strip()
    if not text:
        return "OTHER"

    filled_prompt = classification_prompt.replace("[DOCUMENTS]", f"- {text}")
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
import fasttext
def detect_language(text):
    """Detect language using fasttext model."""
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