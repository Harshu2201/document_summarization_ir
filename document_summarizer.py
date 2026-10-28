"""
Document Summarization System
--------------------------------
Final-year Computer Engineering mini-project demonstrating Information
Retrieval (IR) based extractive document summarization.

Core idea:
1. Read a TXT or PDF document.
2. Split the document into sentences.
3. Represent each sentence using TF-IDF.
4. Give every sentence an importance score based on the TF-IDF weights
   of the terms it contains.
5. Select the highest-scoring sentences.
6. Restore their original order and return about 35% of the sentences.

This is extractive summarization: the program does not generate new text.
It selects important sentences directly from the source document.
"""

from pathlib import Path
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
TARGET_RATIO = 0.35


class SummarizationError(Exception):
    """Custom exception for expected summarization/input errors."""


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 plain-text file.

    UTF-8 is used because it supports normal English text as well as many
    international characters. errors='replace' prevents a single malformed
    byte from crashing the complete prototype.
    """
    path = Path(file_path)

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SummarizationError(f"Could not read text file: {exc}") from exc


def read_pdf_file(file_path: str) -> str:
    """Extract text from every page of a PDF using pypdf.

    PDF files store layout information rather than clean paragraphs, so the
    extracted text may contain extra whitespace. That is acceptable for this
    educational prototype because the IR stage only needs usable text.
    """
    if PdfReader is None:
        raise SummarizationError(
            "PDF support requires pypdf. Install dependencies with: "
            "pip install -r requirements.txt"
        )

    try:
        reader = PdfReader(file_path)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        return "\n".join(pages)

    except Exception as exc:
        raise SummarizationError(f"Could not read PDF file: {exc}") from exc


def read_docx_file(file_path: str) -> str:
    """Extract readable paragraph text from a Microsoft Word .docx file.

    The DOCX format is a ZIP/XML document rather than a plain-text file, so
    python-docx is used to read paragraph content. Tables are also included
    when they contain text, which makes the upload feature more useful for
    reports and notes.
    """
    if Document is None:
        raise SummarizationError(
            "DOCX support requires python-docx. Install dependencies with: "
            "pip install -r requirements.txt"
        )

    try:
        document = Document(file_path)
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(". ".join(cells))

        return "\n".join(parts)
    except Exception as exc:
        raise SummarizationError(f"Could not read DOCX file: {exc}") from exc


def load_document(file_path: str) -> str:
    """Load a supported document based on its file extension.

    This function is the input layer of the system. Keeping file handling
    separate from summarization makes the IR algorithm reusable.
    """
    path = Path(file_path)

    if not path.exists():
        raise SummarizationError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise SummarizationError(f"Path is not a file: {file_path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        text = read_text_file(file_path)
    elif extension == ".pdf":
        text = read_pdf_file(file_path)
    elif extension == ".docx":
        text = read_docx_file(file_path)
    else:
        raise SummarizationError(
            f"Unsupported format '{extension}'. Supported formats: .txt, .pdf, .docx"
        )

    text = clean_text(text)

    if not text:
        raise SummarizationError("The document is empty or contains no readable text.")

    return text


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving sentence punctuation.

    We do not aggressively remove punctuation because '.', '?' and '!' help
    the simple sentence tokenizer identify sentence boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight rule-based tokenizer.

    NLTK's Punkt tokenizer is excellent, but it normally needs an additional
    downloaded model. For a small offline student project, a regular
    expression keeps installation simple and avoids external downloads.

    This tokenizer is intentionally simple; abbreviations such as 'Dr.'
    can produce imperfect boundaries.
    """
    text = clean_text(text)

    if not text:
        return []

    # Split after ., ! or ? followed by whitespace and a likely new sentence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)

    # Remove accidental blank entries.
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _sentence_score(tfidf_row) -> float:
    """Calculate the importance of one sentence.

    TF-IDF gives larger values to terms that are frequent in the current
    sentence/document but less common across the sentence collection.

    We use the sum of TF-IDF weights as a simple relevance score. Dividing
    by the number of terms prevents very long sentences from automatically
    winning only because they contain more words.
    """
    values = tfidf_row.data

    if len(values) == 0:
        return 0.0

    return float(values.sum() / len(values))


def rank_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    """Rank sentences from most important to least important using TF-IDF.

    The sentence list is treated like a small document collection:
    each sentence is a 'document' in the vector space model.

    Returns:
        [(original_sentence_index, score), ...] sorted by descending score.
    """
    if len(sentences) < 2:
        return [(0, 1.0)] if sentences else []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]*\b",
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Happens when the document contains only stop words/punctuation.
        return [(index, 0.0) for index in range(len(sentences))]

    scored = []
    for index in range(matrix.shape[0]):
        score = _sentence_score(matrix.getrow(index))
        scored.append((index, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def summarize_text(text: str, ratio: float = TARGET_RATIO) -> str:
    """Generate an extractive summary from raw text.

    The default ratio is 35%, which satisfies the project's requested
    30-40% concise-summary range.

    For very short documents, at least one sentence is returned. For a
    two-sentence document, one sentence is normally enough; for larger
    documents, approximately 35% of the sentences are selected.
    """
    if not text or not text.strip():
        raise SummarizationError("Cannot summarize empty text.")

    if not 0.30 <= ratio <= 0.40:
        raise ValueError("Summary ratio must be between 0.30 and 0.40.")

    sentences = split_sentences(text)

    if not sentences:
        raise SummarizationError("No sentences could be identified.")

    if len(sentences) == 1:
        return sentences[0]

    # Round to the nearest sentence while guaranteeing at least one.
    summary_count = max(1, round(len(sentences) * ratio))

    ranked = rank_sentences(sentences)
    selected_indices = {index for index, _score in ranked[:summary_count]}

    # Ranking decides importance; original order improves readability.
    summary_sentences = [
        sentence
        for index, sentence in enumerate(sentences)
        if index in selected_indices
    ]

    return " ".join(summary_sentences)


def summarize_file(file_path: str, ratio: float = TARGET_RATIO) -> Tuple[str, str]:
    """Load and summarize a supported file.

    Returns:
        (cleaned_original_text, summary)
    """
    text = load_document(file_path)

    # Very short input is not useful for aggressive summarization.
    # Returning the original text is more meaningful than deleting content.
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text, text

    return text, summarize_text(text, ratio=ratio)


def format_result(file_path: str, summary: str) -> str:
    """Create a readable command-line result."""
    return (
        "\n" + "=" * 70 + "\n"
        "DOCUMENT SUMMARIZATION RESULT\n"
        + "=" * 70 + "\n"
        f"Input : {file_path}\n"
        f"Summary ratio : {TARGET_RATIO:.0%}\n\n"
        "SUMMARY\n"
        + "-" * 70 + "\n"
        + summary
        + "\n"
        + "=" * 70
    )


if __name__ == "__main__":
    # Small standalone test so the core module can be executed directly.
    demo = (
        "Information Retrieval is the process of finding useful information "
        "from a large collection of data. Search engines are a common example "
        "of information retrieval systems. TF-IDF is a classical technique "
        "used to estimate how important a term is in a document. It gives "
        "higher weight to terms that are important in one document but not "
        "common across all documents. Extractive summarization can use these "
        "weights to rank sentences. The highest-ranked sentences can then be "
        "combined into a shorter summary while keeping the original wording."
    )
    print(summarize_text(demo))
