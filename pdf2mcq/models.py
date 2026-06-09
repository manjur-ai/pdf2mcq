from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class ContentBlock:
    type: str
    content: str
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MCQQuestion:
    question_html: str
    options: List[str]
    answers: List[int]
    multi: bool
    marks: float
    negative_marks: float
    difficulty: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "question_html": self.question_html,
            "options": self.options,
            "answers": self.answers,
            "multi": self.multi,
            "marks": self.marks,
            "negative_marks": self.negative_marks,
            "difficulty": self.difficulty,
            "explanation": self.explanation,
        }

    def to_pretty_str(self, number: int = 1) -> str:
        multi_tag = " [MULTI]" if self.multi else ""
        lines = [
            f"Q{number}. [{self.difficulty.upper()}]{multi_tag} {self.question_html}",
            f"  Marks: +{self.marks} / -{self.negative_marks}",
            "",
        ]
        for i, opt in enumerate(self.options):
            marker = "\u2713" if i in self.answers else " "
            lines.append(f"  {marker} {chr(65+i)}) {opt}")
        if self.explanation:
            lines += ["", f"  Explanation: {self.explanation}"]
        return "\n".join(lines)


@dataclass
class MCQSet:
    source_url: Optional[str]
    page_title: str
    questions: List[MCQQuestion]
    total_questions: int
    content_summary: str
    total_exam_time: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_exam_time": self.total_exam_time,
            "questions": [q.to_dict() for q in self.questions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_pretty_str(self) -> str:
        lines = [
            f"{'='*60}",
            f"MCQ Set  : {self.page_title}",
            f"Source   : {self.source_url or 'N/A'}",
            f"Questions: {self.total_questions}  |  Exam time: {self.total_exam_time} min",
            f"Summary  : {self.content_summary}",
            f"{'='*60}",
            "",
        ]
        for i, q in enumerate(self.questions, 1):
            lines.append(q.to_pretty_str(i))
            lines.append("")
        return "\n".join(lines)

    def filter_by_difficulty(self, difficulty: str) -> "MCQSet":
        filtered = [q for q in self.questions if q.difficulty.lower() == difficulty.lower()]
        return MCQSet(
            source_url=self.source_url,
            page_title=self.page_title,
            questions=filtered,
            total_questions=len(filtered),
            content_summary=self.content_summary,
            total_exam_time=len(filtered) * 2,
            metadata=self.metadata,
        )
