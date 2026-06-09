from __future__ import annotations

import base64
import io
import os
from typing import List, Optional


_PIL_AVAILABLE = False
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    pass


_DEFAULT_VISION_MODEL = "google/gemini-2.5-flash-lite"
_FREE_VISION_MODEL = "google/gemma-3-12b-it"

_DEFAULT_OCR_PRIORITY = [
    "google/gemini-2.5-flash-lite",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "openai/gpt-4o",
    "pytesseract",
]
_OCR_MODELS_ENV_VAR = "PDF2MCQ_OCR_MODELS"


def _ocr_vision_api(
    image_bytes_list: List[bytes],
    model: str = _DEFAULT_VISION_MODEL,
    api_key: str = "",
    provider: str = "openrouter",
    max_tokens: int = 4096,
) -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No API key for vision API. Pass api_key= or set OPENROUTER_API_KEY env var."
        )

    if provider == "openrouter":
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        client = openai.OpenAI(api_key=api_key)

    content: list = [
        {
            "type": "text",
            "text": (
                "You are an OCR tool. Read the text from this image. "
                "Preserve headings, paragraphs, bullet points, and list items. "
                "If the image contains multiple boxes, dialogs, or columns, "
                "preserve the order as a human would read them naturally. "
                "For book scans, extract only the main page content and "
                "ignore partly visible pages, overlapping pages, "
                "handwritten notes, and any side objects or artifacts. "
                "If the image contains figures, diagrams, or charts, "
                "describe each one concisely. "
                "Output plain text only, no markdown formatting, "
                "no explanations, no commentary."
            ),
        }
    ]
    for img_bytes in image_bytes_list:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _ocr_vision_with_fallback(
    image_bytes_list: List[bytes],
    primary_model: str = _DEFAULT_VISION_MODEL,
    free_model: str = _FREE_VISION_MODEL,
    api_key: str = "",
    provider: str = "openrouter",
    fallback_to_tesseract: bool = True,
    tesseract_lang: str = "eng",
) -> str:
    if primary_model:
        try:
            return _ocr_vision_api(
                image_bytes_list, model=primary_model,
                api_key=api_key, provider=provider,
            )
        except Exception as e:
            err_msg = str(e)
            no_balance = any(
                kw in err_msg.lower()
                for kw in ("insufficient", "balance", "quota", "credits", "402", "payment")
            )
            if no_balance:
                print(f"  [pdf2mcq] \u26a0 {primary_model}: insufficient balance")
            else:
                print(f"  [pdf2mcq] \u26a0 {primary_model} failed: {err_msg[:120]}")

    if free_model and free_model != primary_model:
        try:
            return _ocr_vision_api(
                image_bytes_list, model=free_model,
                api_key=api_key, provider=provider,
            )
        except Exception as e:
            print(f"  [pdf2mcq] \u26a0 {free_model} fallback failed: {str(e)[:120]}")

    if fallback_to_tesseract:
        try:
            import pytesseract
            from PIL import Image

            texts = []
            for img_bytes in image_bytes_list:
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img, lang=tesseract_lang)
                if text.strip():
                    texts.append(text.strip())
            if texts:
                print(f"  [pdf2mcq] \u2713 pytesseract fallback: {sum(len(t) for t in texts)} chars")
                return "\n\n".join(texts)
        except Exception as e:
            print(f"  [pdf2mcq] \u26a0 pytesseract fallback failed: {str(e)[:120]}")

    return ""


def _ocr_pytesseract(image_bytes: bytes, lang: str = "eng") -> str:
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is required for image OCR.\n"
            "Install with:  pip install pytesseract Pillow\n"
            "Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract"
        )
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()
