# pdf2mcq

Convert PDF files — text PDFs, scanned books, mixed documents — into high-quality MCQ questions using AI.

Built on top of **html2mcq**'s PDF pipeline, extracted as a standalone library focused purely on PDF-to-MCQ generation.

---

## Features

- **Smart PDF detection** — automatically detects text PDFs, scanned PDFs, and mixed documents
- **Text PDFs** — fast extraction via PyMuPDF with chunking at sentence boundaries
- **Scanned PDFs** — renders pages as images → vision API OCR (or pytesseract fallback)
- **Mixed PDFs** — text pages via PyMuPDF + scanned pages via OCR, combined intelligently
- **Multiple AI providers:** OpenRouter, Anthropic, OpenAI, Ollama
- **Auto model failover** for MCQ generation
- **CLI & Python API**

---

## Quick Start

## Quick Start

### CLI

```bash
# Single PDF
pdf2mcq --pdf-path textbook.pdf -n 10

# Multiple PDF URLs
pdf2mcq --pdf-url https://example.com/chapter1.pdf --pdf-url https://example.com/chapter2.pdf

# Scan a folder of PDFs
pdf2mcq --pdf-folder ./textbooks/

# Output as JSON
pdf2mcq --pdf-path notes.pdf -o questions.json --format json

# Use vision model directly (skip OCR)
pdf2mcq --pdf-folder ./slides/ --method images2mcq

# Override OCR model per call
pdf2mcq --pdf-path scanned-doc.pdf --ocr-model "google/gemini-2.5-flash-lite"

# Save OCR text to file
pdf2mcq --pdf-path textbook.pdf --save-ocr-path ocr_output.txt

# Custom instructions
pdf2mcq --pdf-path notes.pdf -i "Focus on mathematical derivations"

# Difficulty mix and topic focus
pdf2mcq --pdf-path textbook.pdf --difficulty "40% easy, 40% medium, 20% hard" --topics calculus algebra

# Page range (only process specific pages)
pdf2mcq --pdf-url https://example.com/textbook.pdf --pages "1-10,15,20-25"

# Show progress bar during MCQ generation
pdf2mcq --pdf-folder ./textbooks/ --progress

# Local Ollama
pdf2mcq --pdf-path notes.pdf --provider ollama --mcq-model qwen2.5:7b
```

### Python API

```python
from pdf2mcq import PDFMCQGenerator

gen = PDFMCQGenerator(
    api_key="sk-or-v1-...",
    provider="openrouter",
    mcq_model="google/gemini-2.5-flash-lite",
)

# From local PDF
mcq = gen.from_pdf_paths("textbook.pdf", n=5)
print(mcq.to_pretty_str())

# From URL
mcq = gen.from_pdf_urls("https://example.com/notes.pdf", n=3)
print(mcq.to_json())

# Multiple PDFs
mcq = gen.from_pdf_paths(["chapter1.pdf", "chapter2.pdf", "chapter3.pdf"])

# Page range (only process specific pages)
mcq = gen.from_pdf_urls("https://example.com/textbook.pdf", n=10, pages="1-10,15,20-25")

# Show progress bar
mcq = gen.from_pdf_paths("textbook.pdf", n=10, show_progress=True)
```

### Custom Instructions & Overrides

```python
mcq = gen.from_pdf_paths(
    "lecture-notes.pdf",
    n=10,
    difficulty_mix="50% easy, 50% hard",
    focus_topics=["machine learning", "neural networks"],
    custom_instructions="Focus on mathematical derivations",
    ocr_model="google/gemini-2.5-flash-lite",  # per-call override
    mcq_model="openai/gpt-4o",                  # per-call override
)
```

### Vision Direct Method

```python
gen = PDFMCQGenerator(
    api_key="sk-or-v1-...",
    method="images2mcq",  # send PDF pages as images directly to vision model
)
mcq = gen.from_pdf_paths("scanned-textbook.pdf", n=10)
```

### Auto Model Selection

```python
gen = PDFMCQGenerator(
    api_key="sk-or-v1-...",
    mcq_model="auto",
    mcq_model_list=[
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-31b-it:free",
    ],
)
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Default API key for OpenRouter |
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `OPENAI_API_KEY` | API key for OpenAI |
| `PDF2MCQ_MCQ_MODELS` | Comma-separated MCQ model priority list for `mcq_model="auto"` |
| `PDF2MCQ_OCR_MODELS` | Comma-separated OCR model priority list for scanned PDFs |

---

## Output Format

```python
# Pretty-print
print(mcq.to_pretty_str())

# JSON
print(mcq.to_json())
# {
#   "total_exam_time": 20,
#   "questions": [
#     {
#       "question_html": "What is gradient descent?",
#       "options": ["...", "...", "...", "..."],
#       "answers": [0],
#       "multi": false,
#       "marks": 1.0,
#       "negative_marks": 0.25,
#       "difficulty": "easy",
#       "explaination": "..."
#     }
#   ]
# }
```

---

## Installation

```bash
pip install pdf2mcq
```

Requires **PyMuPDF** (fitz) — installed automatically as a dependency.

For scanned PDF OCR, also install [Tesseract](https://github.com/tesseract-ocr/tesseract).

---

## License

MIT
