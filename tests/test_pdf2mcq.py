import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pdf2mcq.models import MCQQuestion, MCQSet, ContentBlock
from pdf2mcq.prompts import build_system_prompt, build_user_prompt
from pdf2mcq.image_ocr import (
    _ocr_vision_api,
    _ocr_pytesseract,
    _ocr_vision_with_fallback,
    _PIL_AVAILABLE,
)


class TestModels:
    def test_question_to_dict_schema(self):
        q = MCQQuestion(
            question_html="<b>What is 2+2?</b>",
            options=["3", "4", "5", "6"],
            answers=[1],
            multi=False,
            marks=1.0,
            negative_marks=0.25,
            difficulty="easy",
            explaination="Basic addition.",
        )
        d = q.to_dict()
        assert d["question_html"] == "<b>What is 2+2?</b>"
        assert d["answers"] == [1]
        assert d["multi"] is False

    def test_mcqset_to_json_schema(self):
        q = MCQQuestion(
            question_html="Q", options=["A", "B", "C", "D"],
            answers=[0], multi=False,
            marks=1.0, negative_marks=0.25,
            difficulty="easy", explaination="",
        )
        s = MCQSet(
            source_url="test", page_title="Test",
            questions=[q, q], total_questions=2,
            content_summary="2 pdf chunks",
        )
        data = json.loads(s.to_json())
        assert "total_exam_time" in data
        assert len(data["questions"]) == 2

    def test_filter_by_difficulty(self):
        q1 = MCQQuestion("Q1", ["A","B","C","D"],[0],False,1,0.25,"easy","")
        q2 = MCQQuestion("Q2", ["A","B","C","D"],[0],False,1,0.25,"hard","")
        s = MCQSet("u","T",[q1,q2],2,"2 chunks")
        filtered = s.filter_by_difficulty("easy")
        assert len(filtered.questions) == 1


class TestPrompts:
    def test_system_prompt_has_schema(self):
        prompt = build_system_prompt()
        assert "question_html" in prompt
        assert "options" in prompt

    def test_user_prompt_includes_text_and_code(self):
        blocks = [
            ContentBlock(type="pdf_text", content="PDF extracted content here"),
            ContentBlock(type="code", content="print('hello')", metadata={"language": "python"}),
        ]
        prompt = build_user_prompt(blocks, n=5)
        assert "EXTRACTED CONTENT" in prompt
        assert "PDF extracted content" in prompt
        assert "CODE" in prompt
        assert "python" in prompt

    def test_user_prompt_n999(self):
        blocks = [ContentBlock(type="pdf_text", content="Content")]
        prompt = build_user_prompt(blocks, n=999)
        assert "as many" in prompt.lower() or "cover all" in prompt.lower()


class TestPDFExtractor:
    def test_is_pdf_url(self):
        from pdf2mcq.pdf import _is_pdf_url
        assert _is_pdf_url("https://example.com/doc.pdf") is True
        assert _is_pdf_url("https://example.com/doc.pdf?download=1") is True
        assert _is_pdf_url("https://example.com/doc.html") is False

    def test_fetch_bytes_fails_gracefully(self):
        from pdf2mcq.pdf import _fetch_bytes
        with pytest.raises(Exception):
            _fetch_bytes("https://nonexistent.example/file.pdf")

    def test_chunk_text_simple(self):
        from pdf2mcq.pdf import _chunk_text
        text = "A. " * 2000
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        assert all(len(c) <= 550 for c in chunks)

    def test_chunk_text_short(self):
        from pdf2mcq.pdf import _chunk_text
        chunks = _chunk_text("Short text.", chunk_size=500)
        assert len(chunks) == 1

    def test_detect_scan_type_valid_pdf(self):
        from pdf2mcq.pdf import PDFExtractor
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(fitz.Point(50, 50), "Hello world " * 50)
        b = doc.tobytes()
        doc.close()
        extractor = PDFExtractor()
        result = extractor.detect_scan_type(b)
        assert result == "text"


class TestPDFMCQGenerator:
    def test_missing_api_key_raises(self):
        from pdf2mcq.generator import PDFMCQGenerator
        with pytest.raises(ValueError, match="No API key"):
            PDFMCQGenerator(api_key="")

    def test_ollama_no_api_key_required(self):
        from pdf2mcq.generator import PDFMCQGenerator
        gen = PDFMCQGenerator(provider="ollama")
        assert gen.provider == "ollama"

    def test_parse_response_handles_markdown_fences(self):
        from pdf2mcq.generator import PDFMCQGenerator
        raw = '```json\n[{"question_html":"Q","options":["A","B","C","D"],"answers":[0],"difficulty":"easy","explaination":""}]\n```'
        qs = PDFMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1

    def test_parse_response_handles_single_int_answer(self):
        from pdf2mcq.generator import PDFMCQGenerator
        raw = '[{"question_html":"Q","options":["A","B","C","D"],"answers":2,"difficulty":"medium","explaination":""}]'
        qs = PDFMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1
        assert qs[0].answers == [2]

    def test_parse_response_skips_malformed(self):
        from pdf2mcq.generator import PDFMCQGenerator
        raw = '[null, {"question_html":"Q","options":["A","B","C","D"],"answers":[0],"difficulty":"easy","explaination":""}]'
        qs = PDFMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1

    def test_parse_response_invalid_json_raises(self):
        from pdf2mcq.generator import PDFMCQGenerator
        with pytest.raises(ValueError, match="non-JSON"):
            PDFMCQGenerator._parse_response(None, "not json at all")

    def test_get_mcq_models_default(self):
        from pdf2mcq.generator import PDFMCQGenerator
        models = PDFMCQGenerator.get_mcq_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_set_and_get_mcq_models(self):
        from pdf2mcq.generator import PDFMCQGenerator
        PDFMCQGenerator.set_mcq_models("model-a,model-b")
        models = PDFMCQGenerator.get_mcq_models()
        assert models == ["model-a", "model-b"]

    def test_set_api_key_sets_when_empty(self):
        from pdf2mcq.generator import PDFMCQGenerator
        os.environ.pop("OPENROUTER_API_KEY", None)
        PDFMCQGenerator.set_api_key("openrouter", "sk-test-key")
        assert os.environ.get("OPENROUTER_API_KEY") == "sk-test-key"
        del os.environ["OPENROUTER_API_KEY"]

    def test_set_api_key_ignores_when_already_set(self, monkeypatch):
        from pdf2mcq.generator import PDFMCQGenerator
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-existing")
        PDFMCQGenerator.set_api_key("openrouter", "sk-new-key")
        assert os.environ["OPENROUTER_API_KEY"] == "sk-existing"

    def test__build_summary(self):
        from pdf2mcq.generator import PDFMCQGenerator
        blocks = [
            ContentBlock(type="pdf_text", content="text1"),
            ContentBlock(type="code", content="code"),
        ]
        summary = PDFMCQGenerator._build_summary(blocks)
        assert "1 pdf_text" in summary
        assert "1 code" in summary

    def test_from_pdf_paths_empty_pdf_raises(self, tmp_path):
        from pdf2mcq.generator import PDFMCQGenerator
        p = tmp_path / "empty.pdf"
        p.write_text("not a pdf")
        gen = PDFMCQGenerator(api_key="sk-test")
        with pytest.raises((ValueError, Exception)):
            gen.from_pdf_paths(str(p), n=1)

    def test_from_pdf_urls_empty_list_raises(self):
        from pdf2mcq.generator import PDFMCQGenerator
        gen = PDFMCQGenerator(api_key="sk-test")
        with pytest.raises(ValueError, match="No text"):
            gen.from_pdf_urls([], n=1)


class TestCLI:
    def test_cli_version(self, monkeypatch):
        import sys
        from pdf2mcq import cli
        monkeypatch.setattr(sys, "argv", ["pdf2mcq", "--version"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0

    def test_cli_no_input_shows_error(self, monkeypatch):
        import sys
        from pdf2mcq import cli
        monkeypatch.setattr(sys, "argv", ["pdf2mcq"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_cli_pdf_folder_not_found(self, monkeypatch):
        import sys
        from pdf2mcq import cli
        monkeypatch.setattr(sys, "argv", ["pdf2mcq", "--pdf-folder", "/nonexistent",
                                          "--api-key", "sk-test"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_cli_pdf_folder(self, tmp_path, monkeypatch):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "chapter1.pdf").write_text("dummy")
        (pdf_dir / "chapter2.pdf").write_text("dummy")
        monkeypatch.setattr(sys, "argv", ["pdf2mcq", "--pdf-folder", str(pdf_dir),
                                          "-n", "1", "--api-key", "sk-test"])
        from pdf2mcq.generator import PDFMCQGenerator
        from pdf2mcq.models import MCQQuestion
        orig = PDFMCQGenerator.from_pdf_paths
        called = []
        def mock_method(self, paths, **kw):
            called.extend(paths)
            q = MCQQuestion("Q", ["A","B","C","D"],[0],False,1,0.25,"easy","")
            return MCQSet("test", "PDFs", [q], 1, "")
        try:
            PDFMCQGenerator.from_pdf_paths = mock_method
            from pdf2mcq import cli
            cli.main()
        finally:
            PDFMCQGenerator.from_pdf_paths = orig
        assert len(called) == 2

    def test_cli_env_var_api_key(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv", ["pdf2mcq", "--pdf-url", "https://example.com/doc.pdf", "-n", "1"])
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-test")
        from pdf2mcq.generator import PDFMCQGenerator
        from pdf2mcq.models import MCQQuestion
        orig = PDFMCQGenerator.from_pdf_urls
        called = []
        def mock_method(self, urls, **kw):
            called.extend(urls)
            q = MCQQuestion("Q", ["A","B","C","D"],[0],False,1,0.25,"easy","")
            return MCQSet("test", "PDFs", [q], 1, "")
        try:
            PDFMCQGenerator.from_pdf_urls = mock_method
            from pdf2mcq import cli
            cli.main()
        finally:
            PDFMCQGenerator.from_pdf_urls = orig
        assert len(called) == 1
        assert "example.com/doc.pdf" in called[0]
