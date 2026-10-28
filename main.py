"""
Command-line interface for the Document Summarization System.

Examples:
    python main.py sample_documents/sample.txt
    python main.py my_document.pdf --ratio 0.30
    python main.py my_document.txt --output summary.txt
"""

import argparse
from pathlib import Path

from document_summarizer import (
    TARGET_RATIO,
    SummarizationError,
    summarize_file,
    format_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IR-based extractive document summarization using TF-IDF."
    )
    parser.add_argument(
        "file",
        help="Path to a .txt or .pdf document",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=TARGET_RATIO,
        help="Summary ratio between 0.30 and 0.40 (default: 0.35)",
    )
    parser.add_argument(
        "--output",
        help="Optional path where only the summary will be saved",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        _original, summary = summarize_file(args.file, ratio=args.ratio)

        if args.output:
            Path(args.output).write_text(summary, encoding="utf-8")
            print(f"Summary saved to: {args.output}")
        else:
            print(format_result(args.file, summary))

    except (SummarizationError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
