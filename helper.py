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
- Identification (D.I): Identifying problems or defining an issue to be solved.
- Goal Setting and Requirement Clarification (D.G): Setting learning goals or clarifying course requirements.

## Seeking
- Search (S.S): Obtaining information from the chatbot.
- Select (S.SL): Extracting key points or requesting summaries.
- Evaluation of Information Quality (S.EQ): Checking the credibility, reliability, or sources of information.

## Engaging
- Review (E.RV): Revisiting materials or verifying correctness.
- Organise (E.O): Structuring content, categorizing information.
- Reformatting and Reworking (E.RF): Transforming content into a different format
  (e.g., translation, diagrams, alternative explanations).
- Rehearse (E.RH): Practicing learned concepts, generating quiz questions.

## Reflecting
- Task Evaluation (R.ET): Assessing quality of work or readiness to move forward.
- Self Evaluation (R.ES): Checking for learning gaps or self-reflection.

## Other
- If an entry cannot be clearly assigned to any of these categories, assign it to the OTHER category.

You are given a single topic consisting of multiple user queries.

Representative user queries:
[DOCUMENTS]

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


def classify_prompt(doc: Optional[str], model: str = "gpt-3.5-turbo") -> str:
    """Classify a single prompt and return the action code."""
    text = (doc or "").strip()
    if not text:
        return "OTHER"

    filled_prompt = classification_prompt.replace("[DOCUMENTS]", f"- {text}")
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": filled_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network dependency
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