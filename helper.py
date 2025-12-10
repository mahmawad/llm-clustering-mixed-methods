"""Utility helpers for classifying prompts into predefined categories."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from openai import OpenAI
import toml

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

Topic keywords:
[KEYWORDS]

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


def classify_documents(
    df: pd.DataFrame,
    *,
    text_column: str = "Prompt",
    model: str = "gpt-3.5-turbo",
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Inject each document into the classification prompt and store responses."""

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    result_df = df.copy()
    categories: List[str] = []

    for prompt in result_df[text_column].fillna(""):
        doc = str(prompt).strip()
        if not doc:
            categories.append("OTHER")
            continue

        tokens = [token.strip(" ,.;:!?") for token in doc.split() if token.strip()]
        keyword_block = ", ".join(tokens[:5]) or "General Query"
        doc_block = f"- {doc}"
        filled_prompt = classification_prompt.replace("[KEYWORDS]", keyword_block)
        filled_prompt = filled_prompt.replace("[DOCUMENTS]", doc_block)

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": filled_prompt}],
            )
            categories.append(response.choices[0].message.content.strip())
        except Exception as exc:  # pragma: no cover - network dependency
            print(f"Error classifying prompt: {exc}")
            categories.append("ERROR")

    result_df["Category"] = categories

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(output_path, index=False, sheet_name="Classifications")

    return result_df
