from __future__ import annotations

import io
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .models import ContentBlock


_MIN_MEANINGFUL_CHARS = 100


def _is_pdf_url(url: str) -> bool:
    return bool(re.search(r"\.pdf($|\?|#)", url, re.IGNORECASE))


def _fetch_bytes(url: str, timeout: int = 30, user_agent: str = "pdf2mcq/1.0") -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
    chunks, start = [], 0
    text = text.strip()
    n = len(text)
    while start < n:
        end = start + chunk_size
        if end >= n:
            chunks.append(text[start:].strip())
            break
        boundary = max(
            text.rfind(". ", start, end),
            text.rfind("! ", start, end),
            text.rfind("? ", start, end),
            text.rfind("\n", start, end),
        )
        if boundary > start + chunk_size // 2:
            end = boundary + 1
        else:
            wb = text.rfind(" ", start, end)
            if wb > start:
                end = wb
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return [c for c in chunks if c]


def _parse_page_range(range_str: Optional[str]) -> Optional[List[int]]:
    if not range_str or not range_str.strip():
        return None
    pages: List[int] = []
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start, end = int(start_str.strip()), int(end_str.strip())
                if start < 1 or end < start:
                    raise ValueError
                pages.extend(range(start - 1, end))
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid page range: '{part}'. Use format like '1-10' or '1,3,5'."
                )
        else:
            try:
                p = int(part)
                if p < 1:
                    raise ValueError
                pages.append(p - 1)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid page number: '{part}'. Pages are 1-indexed."
                )
    return sorted(set(pages))


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = len(doc)
    doc.close()
    return n


def _render_specific_pages(pdf_bytes: bytes, page_nums: List[int], max_pages: int = 0) -> List[bytes]:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page_num in enumerate(page_nums):
        if max_pages and i >= max_pages:
            break
        if page_num < len(doc):
            pix = doc[page_num].get_pixmap(dpi=200)
            images.append(pix.tobytes("png"))
    doc.close()
    return images


def _render_pdf_pages_to_pngs(pdf_bytes: bytes, max_pages: int = 0) -> List[bytes]:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        pix = page.get_pixmap(dpi=200)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


class _PyMuPDFBackend:
    name = "pymupdf"

    def __init__(self):
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF extraction.\n"
                "Install with:  pip install pdf2mcq  or  pip install pymupdf"
            )

    def extract(self, pdf_bytes: bytes, source_url: str = "",
                page_numbers: Optional[List[int]] = None) -> Tuple[str, List[Dict]]:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        full_parts = []

        for page_num, page in enumerate(doc, 1):
            if page_numbers is not None and (page_num - 1) not in page_numbers:
                continue
            text = page.get_text("text").strip()
            tables = []
            try:
                page_dict = page.get_text("dict")
                tables = self._extract_tables(page_dict)
            except Exception:
                pass

            pages.append({
                "page": page_num,
                "text": text,
                "tables": tables,
            })
            if text:
                full_parts.append(f"[Page {page_num}]\n{text}")

        doc.close()
        return "\n\n".join(full_parts), pages

    @staticmethod
    def detect_scan_type(pdf_bytes: bytes) -> str:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)
        scanned_pages = 0
        text_pages = 0

        for page in doc:
            text_len = len(page.get_text("text").strip())
            img_count = len(page.get_images())
            if text_len < 10 and img_count >= 1:
                scanned_pages += 1
            elif text_len >= 100:
                text_pages += 1

        doc.close()

        if total == 0:
            return "text"

        scanned_ratio = scanned_pages / total
        text_ratio = text_pages / total

        if scanned_ratio >= 0.8:
            return "scanned"
        if text_ratio >= 0.8:
            return "text"
        if scanned_pages == 0 and text_pages == 0:
            return "text"
        return "mixed"

    @staticmethod
    def _extract_tables(page_dict: dict) -> List[str]:
        lines: Dict[int, List[str]] = {}
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                y = round(line["bbox"][1] / 5) * 5
                spans = [s["text"].strip() for s in line.get("spans", []) if s["text"].strip()]
                if spans:
                    lines.setdefault(y, []).extend(spans)
        table_rows = [
            " | ".join(cells)
            for cells in lines.values()
            if len(cells) >= 3
        ]
        return table_rows


class PDFExtractor:
    def __init__(
        self,
        backend: str = "auto_detect",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        min_meaningful_chars: int = _MIN_MEANINGFUL_CHARS,
        timeout: int = 30,
        user_agent: str = "pdf2mcq/1.0",
        scanned_backend: str = "vision_api",
        scanned_max_pages: int = 50,
        vision_provider: str = "openrouter",
        vision_model: str = "openai/gpt-4o-mini",
        vision_free_model: str = "google/gemma-3-12b-it",
        vision_api_key: str = "",
        ocr_fallback: bool = True,
        ocr_lang: str = "eng",
        ocr_models: Optional[List[str]] = None,
    ):
        self.backend = backend.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_meaningful_chars = min_meaningful_chars
        self.timeout = timeout
        self.user_agent = user_agent
        self.scanned_backend = scanned_backend.lower()
        self.scanned_max_pages = scanned_max_pages

        from .image_ocr import _ocr_vision_api, _ocr_pytesseract, _DEFAULT_OCR_PRIORITY, _OCR_MODELS_ENV_VAR
        if ocr_models:
            self._ocr_models = list(ocr_models)
        else:
            env = os.environ.get(_OCR_MODELS_ENV_VAR, "").strip()
            if env:
                self._ocr_models = [m.strip() for m in env.split(",") if m.strip()]
            else:
                self._ocr_models = list(_DEFAULT_OCR_PRIORITY)

        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_free_model = vision_free_model
        self.vision_api_key = vision_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.ocr_fallback = ocr_fallback
        self.ocr_lang = ocr_lang
        self._primary = self._make_backend(self.backend)

    def from_url(self, url: str,
                 page_numbers: Optional[List[int]] = None) -> List[ContentBlock]:
        print(f"  [pdf2mcq] Downloading PDF: {url}")
        pdf_bytes = _fetch_bytes(url, timeout=self.timeout, user_agent=self.user_agent)
        return self.from_bytes(pdf_bytes, source_url=url, page_numbers=page_numbers)

    def from_bytes(self, pdf_bytes: bytes, source_url: str = "",
                   page_numbers: Optional[List[int]] = None) -> List[ContentBlock]:
        if self.backend == "image":
            return self._extract_scanned(pdf_bytes, source_url, page_numbers=page_numbers)

        if self.backend == "pymupdf":
            full_text, pages, backend_used = self._extract_with_fallback(
                pdf_bytes, source_url, page_numbers=page_numbers)
            if not full_text.strip():
                print(f"  [pdf2mcq] \u26a0 No text extracted from PDF: {source_url}")
                return []
            return self._make_blocks(full_text, pages, backend_used, source_url)

        scan_type = self.detect_scan_type(pdf_bytes)
        print(f"  [pdf2mcq] PDF scan type: {scan_type} ({source_url})")

        if scan_type == "scanned":
            return self._extract_scanned(pdf_bytes, source_url, page_numbers=page_numbers)
        elif scan_type == "mixed":
            blocks = self._extract_mixed(pdf_bytes, source_url, page_numbers=page_numbers)
            if blocks:
                return blocks

        full_text, pages, backend_used = self._extract_with_fallback(
            pdf_bytes, source_url, page_numbers=page_numbers)
        if not full_text.strip():
            print(f"  [pdf2mcq] \u26a0 No text extracted from PDF: {source_url}")
            return []
        return self._make_blocks(full_text, pages, backend_used, source_url)

    def from_path(self, path: str,
                  page_numbers: Optional[List[int]] = None) -> List[ContentBlock]:
        pdf_bytes = Path(path).read_bytes()
        return self.from_bytes(pdf_bytes, source_url=f"file://{path}",
                               page_numbers=page_numbers)

    def detect_scan_type(self, pdf_bytes: bytes) -> str:
        return self._primary.detect_scan_type(pdf_bytes)

    def detect_scan_type_from_path(self, path: str) -> str:
        return self.detect_scan_type(Path(path).read_bytes())

    def enrich_blocks(self, blocks: List[ContentBlock], replace: bool = True) -> List[ContentBlock]:
        enriched: List[ContentBlock] = []
        for block in blocks:
            if block.type == "pdf" and _is_pdf_url(block.content):
                try:
                    pdf_blocks = self.from_url(block.content)
                    if replace:
                        enriched.extend(pdf_blocks)
                    else:
                        enriched.append(block)
                        enriched.extend(pdf_blocks)
                except Exception as e:
                    enriched.append(block)
                    print(f"  [pdf2mcq] \u26a0 Could not extract PDF {block.content}: {e}")
            else:
                enriched.append(block)
        return enriched

    def _extract_scanned(self, pdf_bytes: bytes, source_url: str,
                         page_numbers: Optional[List[int]] = None) -> List[ContentBlock]:
        from .image_ocr import _ocr_vision_api

        page_count = _count_pdf_pages(pdf_bytes)
        if page_numbers is not None:
            page_numbers = [p for p in page_numbers if p < page_count]
            print(f"  [pdf2mcq] Rendering {len(page_numbers)}/{page_count} pages as images for vision OCR...")
            pngs = _render_specific_pages(pdf_bytes, page_numbers, max_pages=self.scanned_max_pages)
        else:
            print(f"  [pdf2mcq] Rendering {page_count} pages as images for vision OCR...")
            pngs = _render_pdf_pages_to_pngs(pdf_bytes, max_pages=self.scanned_max_pages)
        if not pngs:
            return []

        if self.scanned_backend == "auto":
            text = self._ocr_scanned_via_auto(pngs)
        elif self.scanned_backend == "pytesseract":
            text = self._ocr_scanned_via_pytesseract(pngs)
        else:
            try:
                text = _ocr_vision_api(
                    pngs, model=self.scanned_backend,
                    api_key=self.vision_api_key, provider=self.vision_provider,
                )
                if text.strip():
                    pass
            except Exception as e:
                err_msg = str(e)
                no_balance = any(
                    kw in err_msg.lower()
                    for kw in ("insufficient", "balance", "quota", "credits", "402", "payment")
                )
                if no_balance:
                    print(f"  [pdf2mcq] \u26a0 {self.scanned_backend}: insufficient balance")
                else:
                    print(f"  [pdf2mcq] \u26a0 {self.scanned_backend} failed: {err_msg[:120]}")
                fallback = [m for m in self._ocr_models if m != self.scanned_backend]
                if fallback:
                    print(f"  [pdf2mcq] \u2192 falling back to auto (skipping {self.scanned_backend})")
                    text = self._ocr_scanned_via_auto(pngs, models=fallback)
                else:
                    text = ""

        if not text.strip():
            print(f"  [pdf2mcq] \u26a0 No text extracted from scanned PDF: {source_url}")
            return []

        backend_name = f"scanned_{self.scanned_backend}"
        chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
        blocks = []
        for i, chunk in enumerate(chunks):
            blocks.append(ContentBlock(
                type="pdf_text",
                content=chunk,
                metadata={
                    "source_url": source_url,
                    "backend": backend_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "total_pages": page_count,
                    "char_count": len(chunk),
                    "scanned": True,
                },
            ))

        print(
            f"  [pdf2mcq] \u2713 Scanned PDF extracted via {backend_name}: "
            f"{page_count} pages \u2192 {len(blocks)} chunks ({len(text)} chars)"
        )
        return blocks

    def _extract_mixed(self, pdf_bytes: bytes, source_url: str,
                       page_numbers: Optional[List[int]] = None) -> List[ContentBlock]:
        import fitz
        from .image_ocr import _ocr_vision_api

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        scanned_page_nums = []
        text_page_texts = {}
        for i, page in enumerate(doc):
            if page_numbers is not None and i not in page_numbers:
                continue
            text_len = len(page.get_text("text").strip())
            img_count = len(page.get_images())
            if text_len < 10 and img_count >= 1:
                scanned_page_nums.append(i)
            elif text_len >= 10:
                text_page_texts[i] = page.get_text("text").strip()
        total_pages = len(doc)
        doc.close()

        text_content = "\n\n".join(
            f"[Page {n+1}]\n{t}" for n, t in sorted(text_page_texts.items())
        )

        scanned_content = ""
        if scanned_page_nums:
            print(f"  [pdf2mcq] Mixed PDF: {len(text_page_texts)} text pages, "
                  f"{len(scanned_page_nums)} scanned pages \u2192 rendering...")
            pngs = _render_specific_pages(pdf_bytes, scanned_page_nums,
                                           max_pages=self.scanned_max_pages)
            if pngs:
                if self.scanned_backend == "auto":
                    scanned_content = self._ocr_scanned_via_auto(pngs)
                elif self.scanned_backend == "pytesseract":
                    scanned_content = self._ocr_scanned_via_pytesseract(pngs)
                else:
                    try:
                        scanned_content = _ocr_vision_api(
                            pngs, model=self.scanned_backend,
                            api_key=self.vision_api_key, provider=self.vision_provider,
                        )
                        if scanned_content.strip():
                            pass
                    except Exception as e:
                        err_msg = str(e)
                        no_balance = any(
                            kw in err_msg.lower()
                            for kw in ("insufficient", "balance", "quota", "credits", "402", "payment")
                        )
                        if no_balance:
                            print(f"  [pdf2mcq] \u26a0 {self.scanned_backend}: insufficient balance")
                        else:
                            print(f"  [pdf2mcq] \u26a0 {self.scanned_backend} failed: {err_msg[:120]}")
                        fallback = [m for m in self._ocr_models if m != self.scanned_backend]
                        if fallback:
                            print(f"  [pdf2mcq] \u2192 falling back to auto (skipping {self.scanned_backend})")
                            scanned_content = self._ocr_scanned_via_auto(pngs, models=fallback)
                        else:
                            scanned_content = ""

        full_text = text_content
        if scanned_content.strip():
            full_text += "\n\n[Scanned pages OCR]\n" + scanned_content

        if not full_text.strip():
            return []

        backend_name = f"mixed_pymupdf+{self.scanned_backend}"
        chunks = _chunk_text(full_text, self.chunk_size, self.chunk_overlap)
        blocks = []
        for i, chunk in enumerate(chunks):
            blocks.append(ContentBlock(
                type="pdf_text",
                content=chunk,
                metadata={
                    "source_url": source_url,
                    "backend": backend_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "total_pages": total_pages,
                    "char_count": len(chunk),
                    "scanned_pages_ocr": len(scanned_page_nums),
                    "text_pages": len(text_page_texts),
                },
            ))

        print(
            f"  [pdf2mcq] \u2713 Mixed PDF extracted via {backend_name}: "
            f"{total_pages} pages ({len(text_page_texts)} text + "
            f"{len(scanned_page_nums)} scanned) \u2192 {len(blocks)} chunks ({len(full_text)} chars)"
        )
        return blocks

    def _ocr_scanned_via_pytesseract(self, pngs: List[bytes]) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pip install pytesseract Pillow")

        texts = []
        for img_bytes in pngs:
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            if text.strip():
                texts.append(text.strip())
        return "\n\n".join(texts)

    def _ocr_scanned_via_auto(self, pngs: List[bytes],
                                models: Optional[List[str]] = None) -> str:
        from .image_ocr import _ocr_vision_api, _ocr_pytesseract

        for model in models or self._ocr_models:
            if model.lower() == "pytesseract":
                try:
                    texts = []
                    for img_bytes in pngs:
                        text = _ocr_pytesseract(img_bytes, lang=self.ocr_lang)
                        if text.strip():
                            texts.append(text.strip())
                    if texts:
                        result = "\n\n".join(texts)
                        print(f"  [pdf2mcq] \u2713 {model}: {len(result)} chars")
                        return result
                except Exception as e:
                    print(f"  [pdf2mcq] \u26a0 {model} failed: {str(e)[:120]}")
                    continue
            else:
                try:
                    result = _ocr_vision_api(
                        pngs, model=model,
                        api_key=self.vision_api_key, provider=self.vision_provider,
                    )
                    if result:
                        print(f"  [pdf2mcq] \u2713 {model}: {len(result)} chars")
                        return result
                except Exception as e:
                    err_msg = str(e)
                    no_balance = any(
                        kw in err_msg.lower()
                        for kw in ("insufficient", "balance", "quota", "credits", "402", "payment")
                    )
                    if no_balance:
                        print(f"  [pdf2mcq] \u26a0 {model}: insufficient balance")
                    else:
                        print(f"  [pdf2mcq] \u26a0 {model} failed: {err_msg[:120]}")
                    continue
        return ""

    @staticmethod
    def _resolve_ocr_models(ocr_models: Optional[List[str]] = None) -> List[str]:
        if ocr_models:
            return list(ocr_models)
        from .image_ocr import _OCR_MODELS_ENV_VAR, _DEFAULT_OCR_PRIORITY
        env = os.environ.get(_OCR_MODELS_ENV_VAR, "").strip()
        if env:
            return [m.strip() for m in env.split(",") if m.strip()]
        return list(_DEFAULT_OCR_PRIORITY)

    def _make_blocks(self, full_text: str, pages: List[Dict], backend_used: str, source_url: str) -> List[ContentBlock]:
        chunks = _chunk_text(full_text, self.chunk_size, self.chunk_overlap)
        blocks = []
        for i, chunk in enumerate(chunks):
            blocks.append(ContentBlock(
                type="pdf_text",
                content=chunk,
                metadata={
                    "source_url": source_url,
                    "backend": backend_used,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "total_pages": len(pages),
                    "char_count": len(chunk),
                },
            ))

        print(
            f"  [pdf2mcq] \u2713 PDF extracted via {backend_used}: "
            f"{len(pages)} pages \u2192 {len(blocks)} chunks ({len(full_text)} chars)"
        )
        return blocks

    def _make_backend(self, name: str):
        if name in ("pymupdf", "auto_detect", "auto"):
            return _PyMuPDFBackend()
        if name == "image":
            return None
        raise ValueError(
            f"Unknown PDF backend '{name}'. "
            "Choose: auto_detect | pymupdf | image"
        )

    def _extract_with_fallback(self, pdf_bytes: bytes, source_url: str,
                               page_numbers: Optional[List[int]] = None) -> Tuple[str, List[Dict], str]:
        try:
            full_text, pages = self._primary.extract(pdf_bytes, source_url,
                                                     page_numbers=page_numbers)
        except Exception as e:
            print(f"  [pdf2mcq] \u26a0 {self._primary.name} failed: {e}")
            full_text, pages = "", []

        return full_text, pages, self._primary.name
