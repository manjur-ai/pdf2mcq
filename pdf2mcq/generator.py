from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .models import ContentBlock, MCQQuestion, MCQSet
from .prompts import build_system_prompt, build_user_prompt
from .pdf import PDFExtractor, _render_pdf_pages_to_pngs, _count_pdf_pages, _fetch_bytes


class _AnthropicBackend:
    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(self, api_key: str, mcq_model: str):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Missing 'anthropic' package. Install it with: pip install anthropic\n"
                f"  Original error: {e}"
            ) from e
        self.client = anthropic.Anthropic(api_key=api_key)
        self.mcq_model = mcq_model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        msg = self.client.messages.create(
            model=self.mcq_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class _OpenAIBackend:
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, mcq_model: str):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Missing 'openai' package. Install it with: pip install openai\n"
                f"  Original error: {e}"
            ) from e
        self.client = openai.OpenAI(api_key=api_key)
        self.mcq_model = mcq_model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.mcq_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class _OpenRouterBackend:
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"

    def __init__(self, api_key: str, mcq_model: str, site_url: str = "", site_name: str = "pdf2mcq"):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Missing 'openai' package. Install it with: pip install openai\n"
                f"  Original error: {e}"
            ) from e
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )
        self.mcq_model = mcq_model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.mcq_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class _OllamaBackend:
    DEFAULT_MODEL = "qwen2.5:7b"

    def __init__(self, api_key: str, mcq_model: str, ollama_base_url: str = "http://localhost:11434/v1"):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Missing 'openai' package. Install it with: pip install openai\n"
                f"  Original error: {e}"
            ) from e
        self.client = openai.OpenAI(
            api_key=api_key or "ollama",
            base_url=ollama_base_url,
        )
        self.mcq_model = mcq_model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.mcq_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def _make_backend(provider: str, api_key: str, mcq_model: str, **kwargs):
    provider = provider.lower()
    if provider == "anthropic":
        return _AnthropicBackend(api_key, mcq_model)
    if provider == "openai":
        return _OpenAIBackend(api_key, mcq_model)
    if provider == "openrouter":
        return _OpenRouterBackend(api_key, mcq_model)
    if provider == "ollama":
        return _OllamaBackend(api_key, mcq_model, **kwargs)
    raise ValueError(f"Unknown provider '{provider}'. Choose: anthropic | openai | openrouter | ollama")


class _OverrideContext:
    def __init__(self, gen, api_key_override, prompt_log_path,
                 ocr_model=None, mcq_model=None):
        self.gen = gen
        self.api_key = api_key_override
        self.log_path = prompt_log_path
        self.ocr_model = ocr_model
        self.mcq_model = mcq_model
        self._saved = {}

    def __enter__(self):
        if self.api_key is not None:
            self._saved['backend'] = self.gen.backend
            self.gen.backend = _make_backend(
                self.gen.provider, self.api_key, self.gen.mcq_model,
            )
        if self.log_path is not None:
            self._saved['prompt_log_path'] = getattr(self.gen, "prompt_log_path", None)
            self.gen.prompt_log_path = self.log_path
        if self.ocr_model is not None:
            ocr_val = self.ocr_model.lower()
            if hasattr(self.gen, 'pdf_extractor'):
                self._saved['pdf_extractor.scanned_backend'] = self.gen.pdf_extractor.scanned_backend
                self.gen.pdf_extractor.scanned_backend = ocr_val
        if self.mcq_model is not None:
            _DEFAULT_VISION = "google/gemini-2.5-flash-lite"
            _vis_model = self.mcq_model if self.mcq_model not in ("", "auto") else _DEFAULT_VISION
            if hasattr(self.gen, '_vision_model'):
                self._saved['_vision_model'] = self.gen._vision_model
                self.gen._vision_model = _vis_model
            if hasattr(self.gen, 'pdf_extractor') and hasattr(self.gen.pdf_extractor, 'vision_model'):
                self._saved['pdf_extractor.vision_model'] = self.gen.pdf_extractor.vision_model
                self.gen.pdf_extractor.vision_model = _vis_model
            _key = getattr(self.gen, '_resolved_api_key', None) or "ollama"
            self._saved['backend'] = self.gen.backend
            self.gen.backend = _make_backend(self.gen.provider, _key, self.mcq_model)
        return self

    def __exit__(self, *args):
        if 'backend' in self._saved:
            self.gen.backend = self._saved['backend']
        if 'prompt_log_path' in self._saved:
            self.gen.prompt_log_path = self._saved['prompt_log_path']
        if 'pdf_extractor.scanned_backend' in self._saved:
            self.gen.pdf_extractor.scanned_backend = self._saved['pdf_extractor.scanned_backend']
        if 'pdf_extractor.vision_model' in self._saved:
            self.gen.pdf_extractor.vision_model = self._saved['pdf_extractor.vision_model']
        if '_vision_model' in self._saved:
            self.gen._vision_model = self._saved['_vision_model']


class PDFMCQGenerator:
    ENV_KEYS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "ollama": "",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openrouter",
        mcq_model: str = "",
        mcq_model_list: Optional[List[str]] = None,
        ocr_model: str = "pytesseract",
        ocr_models: Optional[List[str]] = None,
        ocr_fallback: bool = True,
        ocr_lang: str = "eng",
        method: str = "twostep",
        batch_size: int = 10,
        max_tokens: int = 4096,
        custom_instructions: Optional[str] = None,
        prompt_log_path: Optional[str] = None,
        save_ocr_path: Optional[str] = None,
        api_key_override: Optional[str] = None,
        extractor_kwargs: Optional[dict] = None,
        pdf_backend: str = "auto_detect",
        pdf_scanned_max_pages: int = 50,
        pdf_chunk_size: int = 1500,
        **backend_kwargs,
    ):
        self.provider = provider.lower()
        self.mcq_model = mcq_model
        self.mcq_model_list = mcq_model_list
        self.method = method.lower()
        if self.method not in ("twostep", "images2mcq"):
            raise ValueError(f"Unknown method '{method}'. Choose: twostep | images2mcq")

        if self.provider == "ollama":
            _key = "ollama"
            if self.mcq_model in ("", "auto"):
                self.mcq_model = _OllamaBackend.DEFAULT_MODEL
        else:
            _key = api_key or os.environ.get(self.ENV_KEYS.get(self.provider, ""), "")
            if not _key:
                raise ValueError(
                    f"No API key supplied. Pass api_key= or set "
                    f"{self.ENV_KEYS.get(self.provider, 'YOUR_API_KEY')} env var."
                )
        self._resolved_api_key = _key
        self.backend = _make_backend(self.provider, _key, self.mcq_model, **backend_kwargs)
        if api_key_override:
            self._resolved_api_key = api_key_override
            self.backend = _make_backend(self.provider, api_key_override, self.mcq_model, **backend_kwargs)
        self.batch_size = max(1, batch_size)
        self.max_tokens = max_tokens

        _DEFAULT_VISION = "google/gemini-2.5-flash-lite"
        if self.mcq_model and self.mcq_model not in ("auto",):
            _vision_model = self.mcq_model
        else:
            _vision_model = _DEFAULT_VISION
        self._vision_model = _vision_model
        self._vision_free_model = "google/gemma-3-12b-it"

        self.pdf_extractor = PDFExtractor(
            backend=pdf_backend,
            chunk_size=pdf_chunk_size,
            scanned_max_pages=pdf_scanned_max_pages,
            scanned_backend=ocr_model,
            vision_provider="openrouter",
            vision_model=_vision_model,
            vision_free_model=self._vision_free_model,
            vision_api_key=_key,
            ocr_fallback=ocr_fallback,
            ocr_lang=ocr_lang,
            ocr_models=ocr_models,
        )

        self.custom_instructions = custom_instructions or ""
        self.prompt_log_path = prompt_log_path
        self.save_ocr_path = save_ocr_path

    def _log_prompt(self, label: str, text: str):
        path = getattr(self, "prompt_log_path", None)
        if path:
            if path in ("-", "stdout"):
                print(f"\n===== {label} =====\n{text}")
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"\n===== {label} =====\n{text}\n")

    def _with_overrides(self, api_key_override: Optional[str] = None,
                        prompt_log_path: Optional[str] = None,
                        ocr_model: Optional[str] = None,
                        mcq_model: Optional[str] = None):
        return _OverrideContext(self, api_key_override, prompt_log_path,
                                ocr_model=ocr_model, mcq_model=mcq_model)

    def _fetch_pdf_render(self, url_or_path: str, source_url: str) -> List[bytes]:
        """Download/read a PDF and render all pages as PNG images."""
        if url_or_path.startswith("file://"):
            pdf_bytes = Path(url_or_path[7:]).read_bytes()
        elif url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            pdf_bytes = _fetch_bytes(url_or_path, timeout=30)
        else:
            pdf_bytes = Path(url_or_path).read_bytes()
        return _render_pdf_pages_to_pngs(pdf_bytes, max_pages=self.pdf_extractor.scanned_max_pages)

    def from_pdf_urls(
        self,
        urls: Union[str, List[str]],
        n: int = 999,
        pdf_title: str = "",
        difficulty_mix: Optional[str] = None,
        focus_topics: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
        api_key_override: Optional[str] = None,
        prompt_log_path: Optional[str] = None,
        ocr_model: Optional[str] = None,
        mcq_model: Optional[str] = None,
    ) -> MCQSet:
        with self._with_overrides(api_key_override, prompt_log_path,
                                  ocr_model=ocr_model, mcq_model=mcq_model):
            if isinstance(urls, str):
                urls = [urls]
            if self.method == "images2mcq":
                all_pngs: List[bytes] = []
                for url in urls:
                    all_pngs.extend(self._fetch_pdf_render(url, url))
                if not all_pngs:
                    raise ValueError("No pages could be rendered from PDF(s)")
                title = pdf_title or (urls[0].split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ").title()
                                      if len(urls) == 1 else "PDFs")
                all_qs = self._vision_mcq_pdf(all_pngs, n=n, page_title=title,
                                              difficulty_mix=difficulty_mix,
                                              focus_topics=focus_topics,
                                              custom_instructions=custom_instructions)
                return self._build_mcq_set(all_qs, n, title, urls[0], [])
            all_blocks: List[ContentBlock] = []
            for url in urls:
                blocks = self.pdf_extractor.from_url(url)
                if not blocks:
                    raise ValueError(f"No text could be extracted from PDF: {url}")
                all_blocks.extend(blocks)
            if not all_blocks:
                raise ValueError("No text could be extracted from any PDF")
            title = pdf_title or (urls[0].split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ").title()
                                  if len(urls) == 1 else "PDFs")
            all_qs, _ = self._generate(
                blocks=all_blocks,
                n=n,
                page_title=title,
                source_url=urls[0],
                difficulty_mix=difficulty_mix,
                focus_topics=focus_topics,
                custom_instructions=custom_instructions,
            )
            return self._build_mcq_set(all_qs, n, title, urls[0], all_blocks)

    def from_pdf_paths(
        self,
        paths: Union[str, List[str]],
        n: int = 999,
        pdf_title: str = "",
        difficulty_mix: Optional[str] = None,
        focus_topics: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
        api_key_override: Optional[str] = None,
        prompt_log_path: Optional[str] = None,
        ocr_model: Optional[str] = None,
        mcq_model: Optional[str] = None,
    ) -> MCQSet:
        with self._with_overrides(api_key_override, prompt_log_path,
                                  ocr_model=ocr_model, mcq_model=mcq_model):
            if isinstance(paths, str):
                paths = [paths]
            if self.method == "images2mcq":
                all_pngs: List[bytes] = []
                for p in paths:
                    all_pngs.extend(self._fetch_pdf_render(str(Path(p).resolve()), f"file://{p}"))
                if not all_pngs:
                    raise ValueError("No pages could be rendered from PDF(s)")
                title = pdf_title or (Path(paths[0]).stem.replace("-", " ").replace("_", " ").title()
                                      if len(paths) == 1 else "PDFs")
                all_qs = self._vision_mcq_pdf(all_pngs, n=n, page_title=title,
                                              difficulty_mix=difficulty_mix,
                                              focus_topics=focus_topics,
                                              custom_instructions=custom_instructions)
                return self._build_mcq_set(all_qs, n, title, f"file://{paths[0]}", [])
            all_blocks: List[ContentBlock] = []
            for p in paths:
                blocks = self.pdf_extractor.from_path(p)
                if not blocks:
                    raise ValueError(f"No text could be extracted from PDF: {p}")
                all_blocks.extend(blocks)
            if not all_blocks:
                raise ValueError("No text could be extracted from any PDF")
            title = pdf_title or (Path(paths[0]).stem.replace("-", " ").replace("_", " ").title()
                                  if len(paths) == 1 else "PDFs")
            all_qs, _ = self._generate(
                blocks=all_blocks,
                n=n,
                page_title=title,
                source_url=f"file://{paths[0]}",
                difficulty_mix=difficulty_mix,
                focus_topics=focus_topics,
                custom_instructions=custom_instructions,
            )
            return self._build_mcq_set(all_qs, n, title, f"file://{paths[0]}", all_blocks)

    from_pdf_url = from_pdf_urls
    from_pdf_path = from_pdf_paths

    def _resolve_instructions(self, per_call: Optional[str]) -> str:
        parts = []
        if self.custom_instructions and self.custom_instructions.strip():
            parts.append(self.custom_instructions.strip())
        if per_call and per_call.strip():
            parts.append(per_call.strip())
        return "\n".join(parts)

    def _build_mcq_set(
        self,
        all_questions: List[MCQQuestion],
        n: int,
        page_title: str,
        source_url: Optional[str],
        blocks: List[ContentBlock],
    ) -> MCQSet:
        all_questions = all_questions[:n]
        summary = self._build_summary(blocks)
        exam_time = max(1, len(all_questions) * 2)
        return MCQSet(
            source_url=source_url,
            page_title=page_title,
            questions=all_questions,
            total_questions=len(all_questions),
            content_summary=summary,
            total_exam_time=exam_time,
            metadata={
                "provider": self.provider,
                "mcq_model": getattr(self.backend, "mcq_model", "unknown"),
                "requested_n": n,
                "content_blocks": len(blocks),
                "content_types": list({b.type for b in blocks}),
            },
        )

    def _vision_mcq_pdf(
        self, pngs: List[bytes], n: int, page_title: str,
        difficulty_mix: Optional[str] = None,
        focus_topics: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> List[MCQQuestion]:
        """Send rendered PDF page images directly to a vision model for MCQ generation."""
        api_key = self.pdf_extractor.vision_api_key
        if not api_key:
            return []

        try:
            import openai
        except ImportError:
            return []

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        instr_parts = [f"Generate {n} MCQ questions based on the content in these PDF pages."]
        if page_title:
            instr_parts.insert(0, f"PAGE TITLE: {page_title}")
        if difficulty_mix:
            instr_parts.append(f"Difficulty distribution: {difficulty_mix}")
        if focus_topics:
            instr_parts.append(f"Focus especially on these topics: {', '.join(focus_topics)}")
        if custom_instructions and custom_instructions.strip():
            instr_parts.append(
                f"\n--- CUSTOM INSTRUCTIONS (highest priority) ---\n"
                f"{custom_instructions.strip()}\n"
                f"--- END CUSTOM INSTRUCTIONS ---"
            )
        instr_parts.append(
            "Return ONLY a JSON array, no markdown. "
            'Each item: {"question_html": "...", "options": ["A","B","C","D"], '
            '"answers": [0], "difficulty": "easy|medium|hard", '
            '"explaination": "..."}'
        )
        content: list = [{"type": "text", "text": "\n".join(instr_parts)}]
        for png in pngs:
            b64 = base64.b64encode(png).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        self._log_prompt("VISION INSTRUCTION",
                          f"Model: {self._vision_model}\n"
                          f"Pages: {len(pngs)}\n"
                          f"Instruction: {content[0]['text']}")

        try:
            resp = client.chat.completions.create(
                model=self._vision_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=8192,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                print("  [pdf2mcq] \u26a0 vision model returned empty response")
                return []
            return self._parse_response(raw)
        except Exception as e:
            print(f"  [pdf2mcq] \u26a0 vision MCQ failed: {e}")
            return []

    @staticmethod
    def _resolve_mcq_model_list(mcq_model_list: Optional[List] = None) -> List[dict]:
        env = os.environ.get("PDF2MCQ_MCQ_MODELS", "").strip()
        if env:
            raw = [m.strip() for m in env.split(",") if m.strip()]
        elif mcq_model_list:
            raw = list(mcq_model_list)
        else:
            raw = [
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "openai/gpt-oss-120b:free",
                "google/gemma-4-31b-it:free",
            ]
        _MAX_TOKENS = {
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": 65536,
            "openai/gpt-oss-120b:free": 131072,
            "openai/gpt-oss-20b:free": 131072,
            "google/gemma-4-31b-it:free": 32768,
            "google/gemma-4-26b-a4b-it:free": 32768,
            "nvidia/nemotron-3-super-120b-a12b:free": 65536,
            "nvidia/nemotron-3-ultra-550b-a55b:free": 65536,
        }
        result = []
        for entry in raw:
            if isinstance(entry, dict):
                result.append(entry)
            else:
                result.append({
                    "model": entry,
                    "max_tokens": _MAX_TOKENS.get(entry, 16384),
                })
        return result

    @staticmethod
    def get_mcq_models() -> list:
        return [entry["model"] for entry in PDFMCQGenerator._resolve_mcq_model_list()]

    @staticmethod
    def set_mcq_models(value: str) -> None:
        os.environ["PDF2MCQ_MCQ_MODELS"] = value

    @staticmethod
    def get_ocr_models() -> list:
        from .pdf import PDFExtractor
        return PDFExtractor._resolve_ocr_models()

    @staticmethod
    def set_ocr_models(value: str) -> None:
        os.environ["PDF2MCQ_OCR_MODELS"] = value

    @staticmethod
    def set_api_key(provider: str, key: str) -> None:
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": "",
        }
        env_var = env_map.get(provider.lower())
        if not env_var:
            return
        if not os.environ.get(env_var, "").strip():
            os.environ[env_var] = key

    def _generate(
        self,
        blocks: List[ContentBlock],
        n: int,
        page_title: str,
        source_url: Optional[str],
        difficulty_mix: Optional[str],
        focus_topics: Optional[List[str]],
        custom_instructions: Optional[str] = None,
    ) -> Tuple[List[MCQQuestion], str]:
        if not blocks:
            return [], ""

        all_questions: List[MCQQuestion] = []
        system_prompt = build_system_prompt()
        remaining = n

        if self.mcq_model == "auto":
            model_list = self._resolve_mcq_model_list(self.mcq_model_list)
            for entry in model_list:
                model_name = entry["model"]
                model_tokens = entry["max_tokens"]
                self.backend.mcq_model = model_name
                est_tokens_per_q = 500
                requested_output = n * est_tokens_per_q + 200
                batch_max_tokens = min(model_tokens, requested_output)
                max_per_call = max(1, batch_max_tokens // est_tokens_per_q)
                batch_n = min(remaining, max_per_call)
                user_prompt = build_user_prompt(
                    blocks=blocks,
                    n=batch_n,
                    difficulty_mix=difficulty_mix,
                    focus_topics=focus_topics,
                    page_title=page_title,
                    custom_instructions=self._resolve_instructions(custom_instructions),
                )
                self._log_prompt("SYSTEM", system_prompt)
                self._log_prompt("USER", user_prompt)
                try:
                    raw = self.backend.complete(system_prompt, user_prompt, batch_max_tokens)
                    batch = self._parse_response(raw)
                except Exception as e:
                    print(f"  [pdf2mcq] \u26a0 MCQ model '{model_name}' failed: {e}")
                    continue
                if batch:
                    all_questions.extend(batch)
                    remaining -= len(batch)
                    print(f"  [pdf2mcq] OK MCQ model '{model_name}' selected "
                          f"({len(batch)} questions, {batch_max_tokens} max_tokens)")
                    break
            else:
                raise RuntimeError(
                    f"All MCQ models in list failed: {[e['model'] for e in model_list]}"
                )
            all_questions = all_questions[:n]
            summary = self._build_summary(blocks)
            return all_questions, summary

        while remaining > 0:
            batch_n = min(remaining, self.batch_size)
            user_prompt = build_user_prompt(
                blocks=blocks,
                n=batch_n,
                difficulty_mix=difficulty_mix,
                focus_topics=focus_topics,
                page_title=page_title,
                custom_instructions=self._resolve_instructions(custom_instructions),
            )
            self._log_prompt("SYSTEM", system_prompt)
            self._log_prompt("USER", user_prompt)
            raw = self.backend.complete(system_prompt, user_prompt, self.max_tokens)
            batch_questions = self._parse_response(raw)
            all_questions.extend(batch_questions)
            remaining -= len(batch_questions)

            if len(batch_questions) == 0:
                break
            if remaining > 0 and len(batch_questions) < batch_n:
                break

        all_questions = all_questions[:n]
        summary = self._build_summary(blocks)
        return all_questions, summary

    def _parse_response(self, raw: str) -> List[MCQQuestion]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"AI returned non-JSON response:\n{raw[:500]}")

        questions = []
        for item in data:
            if item is None:
                continue
            try:
                raw_answers = item.get("answers", item.get("correct_answer", 0))
                if isinstance(raw_answers, int):
                    answers = [raw_answers]
                else:
                    answers = [int(a) for a in raw_answers]

                options = item.get("options", [])
                if not isinstance(options, list):
                    continue
                if not options:
                    continue

                multi = item.get("multi", len(answers) > 1)
                marks = float(item.get("marks", 1))
                negative_marks = float(item.get("negative_marks", 0.0 if multi else 0.25))

                q = MCQQuestion(
                    question_html=item.get("question_html", item.get("question", "")),
                    options=options[:4],
                    answers=answers,
                    multi=bool(multi),
                    marks=marks,
                    negative_marks=negative_marks,
                    difficulty=item.get("difficulty", "medium").lower(),
                    explaination=item.get("explaination", item.get("explanation", "")),
                )
                questions.append(q)
            except (KeyError, TypeError, ValueError):
                continue
        return questions

    @staticmethod
    def _build_summary(blocks: List[ContentBlock]) -> str:
        counts = {}
        for b in blocks:
            counts[b.type] = counts.get(b.type, 0) + 1
        parts = [f"{v} {k}{'s' if v>1 else ''}" for k, v in sorted(counts.items())]
        return "Content: " + ", ".join(parts)
