from __future__ import annotations
from typing import List, Optional
from .models import ContentBlock


_SYSTEM_BASE = """\
You are an expert educator and MCQ JSON generator.

Generate questions only from meaningful educational content provided by the user, including text, code, tables, and extracted PDF content.

For diagrams, figures, charts, and tables, use only information that is visibly shown or clearly illustrated. Ignore advertisements, watermarks, page decorations, and irrelevant content.

Do not use outside knowledge to create extra questions, add new correct facts, or invent information.

General knowledge may be used only to improve question framing and create plausible distractors. Correct answers and explanations must be based only on the provided content.

Return ONLY a valid JSON array. No markdown, no preamble, no extra text.

Schema:
[
  {
    "question_html": "<question text; may use only safe HTML: <b>, <strong>, <em>, <i>, <u>, <code>, <sub>, <sup>, <br>, <ul>, <ol>, <li>>",
    "options": ["<option 0>", "<option 1>", "<option 2>", "<option 3>"],
    "answers": [<0-based correct option index>],
    "multi": <true|false>,
    "marks": <1 if multi=false, 2 if multi=true>,
    "negative_marks": <0.25 if multi=false, 0 if multi=true>,
    "difficulty": "<easy|medium|hard>",
    "explaination": "<brief explanation based only on the content, or empty string>"
  }
]

Rules:
1. Each question must have exactly 4 options.
2. Option indices must be zero-based: 0, 1, 2, 3.
3. Most questions should be single-answer.
4. Use multi-answer questions only when the content clearly supports multiple correct options; target about 20-30% if possible.
5. If "multi" is false:
   - "answers" must contain exactly one index
   - "marks" must be 1
   - "negative_marks" must be 0.25
6. If "multi" is true:
   - "answers" must contain all correct indices
   - "marks" must be 2
   - "negative_marks" must be 0
7. Distractors must be plausible and related to the content.
8. Difficulty should be roughly balanced: easy, medium, hard.
9. Skip clearly wrong facts; do not correct them or make questions from them.
10. Before adding a question, verify that:
    - it is answerable from the content
    - the correct answer index/indices match the options
    - "multi", "marks", and "negative_marks" are consistent
    - "explaination" matches the selected answer(s)
    - no unsupported fact is included
11. Drop any question that fails validation.
"""


def build_system_prompt() -> str:
    return _SYSTEM_BASE


def build_user_prompt(
    blocks: List[ContentBlock],
    n: int,
    difficulty_mix: Optional[str] = None,
    focus_topics: Optional[List[str]] = None,
    page_title: str = "",
    custom_instructions: Optional[str] = None,
) -> str:
    sections: List[str] = []

    if page_title:
        sections.append(f"PAGE TITLE: {page_title}\n")

    text_blocks    = [b for b in blocks if b.type in ("text", "pdf_text")]
    code_blocks    = [b for b in blocks if b.type == "code"]
    table_blocks   = [b for b in blocks if b.type == "table"]

    if text_blocks:
        sections.append("=== EXTRACTED CONTENT ===")
        for i, b in enumerate(text_blocks, 1):
            tag = b.metadata.get("tag", "p")
            sections.append(f"[{tag.upper()} {i}] {b.content}")
        sections.append("")

    if code_blocks:
        sections.append("=== CODE EXAMPLES ===")
        for i, b in enumerate(code_blocks, 1):
            lang = b.metadata.get("language", "")
            sections.append(f"[CODE {i}{' ('+lang+')' if lang else ''}]\n{b.content.strip()}")
        sections.append("")

    if table_blocks:
        sections.append("=== TABLES ===")
        for i, b in enumerate(table_blocks, 1):
            sections.append(f"[TABLE {i}]\n{b.content}")
        sections.append("")

    if n == 999:
        instructions = ["\nBased on the content above, generate as many high-quality MCQ questions as the content supports and cover all distinct valid topics without inventing extra questions."]
    else:
        instructions = [f"\nGenerate exactly {n} MCQ questions based on the content above."]

    if difficulty_mix:
        instructions.append(f"Difficulty distribution: {difficulty_mix}")
    else:
        instructions.append("Mix difficulties: approximately equal easy/medium/hard split.")

    if focus_topics:
        instructions.append(f"Focus especially on these topics: {', '.join(focus_topics)}")

    instructions.append(
        "Use all meaningful educational content: text, code, tables, and diagrams. "
        "Create questions only from visible content; ignore advertisements."
    )
    if custom_instructions and custom_instructions.strip():
        instructions.append(
            f"\n--- CUSTOM INSTRUCTIONS (highest priority, override defaults if needed) ---\n"
            f"{custom_instructions.strip()}\n"
            f"--- END CUSTOM INSTRUCTIONS ---"
        )

    return "\n".join(sections) + "\n" + "\n".join(instructions)
