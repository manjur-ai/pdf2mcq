import argparse
import json
import os
import sys
from pathlib import Path


_PDF_EXTENSION = ".pdf"


def _glob_files(folder: str, extensions: set) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    seen = set()
    unique = []
    for f in sorted(files, key=lambda p: p.name.lower()):
        if f.suffix.lower() in extensions and f.name not in seen:
            seen.add(f.name)
            unique.append(str(f))
    return unique


def _get_api_key(args):
    key = args.api_key or ""
    if key:
        return key
    env_vars = {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": "",
    }
    env_key = env_vars.get(args.provider, "")
    if env_key:
        return os.environ.get(env_key, "")
    return ""


def main():
    parser = argparse.ArgumentParser(
        prog="pdf2mcq",
        description="Convert PDF files (text, scanned, mixed) into MCQ questions using AI.",
    )

    parser.add_argument("--version", action="store_true", help="Show version and exit")

    input_group = parser.add_argument_group("Input sources (at least one required)")
    input_group.add_argument("--pdf-url", metavar="URL", action="append", default=[],
                             help="PDF URL (repeatable: --pdf-url url1 --pdf-url url2)")
    input_group.add_argument("--pdf-path", metavar="FILE", action="append", default=[],
                             help="Local PDF file path (repeatable)")
    input_group.add_argument("--pdf-folder", metavar="DIR", default="",
                             help="Scan folder for PDF files (.pdf)")

    gen_group = parser.add_argument_group("Generation options")
    gen_group.add_argument("-n", "--n", type=int, default=999,
                           help="Number of questions (default: 999 = as many as content supports)")
    gen_group.add_argument("--difficulty", default=None,
                           help='E.g. "30%% easy, 40%% medium, 30%% hard"')
    gen_group.add_argument("--topics", nargs="*", help="Focus topics")
    gen_group.add_argument("--instructions", "-i", default="",
                           help='Custom instructions e.g. "Make answers very close and confusing"')
    gen_group.add_argument("--batch-size", type=int, default=10,
                           help="Questions per API call (default: 10)")

    ai_group = parser.add_argument_group("AI provider")
    ai_group.add_argument("--provider", default="openrouter",
                          choices=["anthropic", "openai", "openrouter", "ollama"],
                          help="AI provider (default: openrouter). Use 'ollama' for local LLM.")
    ai_group.add_argument("--mcq-model", default="",
                          help="MCQ generation model (or 'auto' to try --mcq-models)")
    ai_group.add_argument("--mcq-models", default="",
                          help="Comma-separated priority model list for --mcq-model auto. "
                               "Runtime-reloadable via PDF2MCQ_MCQ_MODELS env var.")
    ai_group.add_argument("--api-key", default="",
                          help="API key. Falls back to OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY env var.")
    ai_group.add_argument("--ollama-base-url", default="http://localhost:11434/v1",
                          help="Ollama API base URL (default: http://localhost:11434/v1). "
                               "Only used when --provider ollama.")

    pdf_group = parser.add_argument_group("PDF processing")
    pdf_group.add_argument("--pdf-backend", default="auto_detect",
                           choices=["auto_detect", "pymupdf", "image"],
                           help="PDF extraction backend (default: auto_detect)")
    pdf_group.add_argument("--scanned-max-pages", type=int, default=50,
                           help="Max pages to OCR for scanned PDFs (default: 50)")
    pdf_group.add_argument("--prompt-log-path", default="",
                           help="Dump prompts to file, or 'stdout' / '-' for terminal")

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output", "-o", default="",
                           help="Output file (.json or .txt). Default: stdout")
    out_group.add_argument("--format", choices=["json", "pretty"], default="pretty",
                           help="Output format (default: pretty)")

    args = parser.parse_args()

    if args.version:
        try:
            from pdf2mcq import __version__
        except ImportError:
            __version__ = "unknown"
        print(f"pdf2mcq v{__version__}")
        sys.exit(0)

    if args.pdf_folder:
        args.pdf_path.extend(_glob_files(args.pdf_folder, {_PDF_EXTENSION}))

    has_input = bool(args.pdf_url or args.pdf_path)
    if not has_input:
        parser.print_help()
        print("\nError: at least one input source is required "
              "(--pdf-url, --pdf-path, --pdf-folder)", file=sys.stderr)
        sys.exit(1)

    try:
        from pdf2mcq import PDFMCQGenerator
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    api_key = _get_api_key(args)
    mcq_model_list = None
    if args.mcq_models:
        mcq_model_list = [m.strip() for m in args.mcq_models.split(",") if m.strip()]

    try:
        gen = PDFMCQGenerator(
            api_key=api_key or None,
            provider=args.provider,
            mcq_model=args.mcq_model,
            mcq_model_list=mcq_model_list,
            batch_size=args.batch_size,
            pdf_backend=args.pdf_backend,
            pdf_scanned_max_pages=args.scanned_max_pages,
            prompt_log_path=args.prompt_log_path or None,
            ollama_base_url=args.ollama_base_url,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    n = args.n
    difficulty = args.difficulty
    topics = args.topics
    instructions = args.instructions or None

    mcq_set = None

    try:
        if args.pdf_url:
            print(f"Fetching PDFs: {args.pdf_url}", file=sys.stderr)
            mcq_set = gen.from_pdf_urls(args.pdf_url, n=n, difficulty_mix=difficulty,
                                        focus_topics=topics, custom_instructions=instructions)
        if args.pdf_path:
            print(f"Reading PDFs: {args.pdf_path}", file=sys.stderr)
            mcq_set = gen.from_pdf_paths(args.pdf_path, n=n, difficulty_mix=difficulty,
                                         focus_topics=topics, custom_instructions=instructions)
    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if mcq_set is None or not mcq_set.questions:
        print("No questions were generated.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        output = mcq_set.to_json()
    else:
        output = mcq_set.to_pretty_str()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {mcq_set.total_questions} questions to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
