# Document Summarization System using Information Retrieval

A final-year Computer Engineering mini-project demonstrating **extractive document summarization** with classical Information Retrieval (IR) techniques.

## 1. Project Objective

The system accepts a plain-text (`.txt`) or PDF (`.pdf`) document and creates a concise summary containing approximately **35% of the original sentences**.

The project demonstrates:

- Document preprocessing
- Sentence tokenization
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Vector-space representation
- Sentence ranking
- Extractive summarization
- Basic input/error handling
- Command-line execution
- Optional Flask web interface

The system does **not** use generative AI or external APIs.

---

## 2. How the IR Approach Works

### Step 1: Document Processing

The selected TXT/PDF file is converted into plain text.

For PDF input, `pypdf` extracts text page by page.

### Step 2: Sentence Tokenization

The text is split into sentences using a lightweight regular-expression tokenizer.

Example:

```text
IR is useful. Search engines use IR.
```

becomes:

```text
[
    "IR is useful.",
    "Search engines use IR."
]
```

This project deliberately uses a simple tokenizer so the project can run without downloading NLTK models.

### Step 3: TF-IDF

Every sentence is treated as a small document.

TF-IDF combines:

**Term Frequency (TF)**  
How frequently a term occurs in a sentence.

**Inverse Document Frequency (IDF)**  
How uncommon that term is across all sentences.

A simplified IDF idea is:

```text
IDF(term) = log(N / DF(term))
```

where:

- `N` = number of sentences
- `DF(term)` = number of sentences containing the term

Common words are less useful for ranking, while terms that distinguish particular sentences receive greater weight.

### Step 4: Sentence Scoring

For each sentence, the system adds its TF-IDF weights and normalizes the total by the number of weighted terms.

Therefore, a sentence containing several important terms receives a higher score.

### Step 5: Sentence Selection

The default target is:

```text
35% of the original sentences
```

This lies inside the requested 30–40% range.

The highest-scoring sentences are selected.

Finally, they are placed back in their original document order. This improves readability because the summary follows the source document's flow instead of appearing in ranking order.

---

## 3. Project Structure

```text
document_summarization_ir/
│
├── document_summarizer.py     # Main IR/NLP logic
├── main.py                    # Command-line interface
├── app.py                     # Optional Flask interface
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
│
└── sample_documents/
    └── sample.txt             # Sample input document
```

---

## 4. Requirements

Recommended:

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses commonly available packages:

- `scikit-learn` — TF-IDF implementation
- `pypdf` — PDF text extraction
- `Flask` — optional web interface

Python standard-library modules are used for file handling, command-line arguments, text cleanup, and sentence splitting.

---

## 5. Run from the Command Line

### TXT file

```bash
python main.py sample_documents/sample.txt
```

### PDF file

```bash
python main.py my_document.pdf
```

### Change the summary ratio

The allowed range is 30%–40%.

```bash
python main.py sample_documents/sample.txt --ratio 0.30
```

or:

```bash
python main.py sample_documents/sample.txt --ratio 0.40
```

### Save summary to a file

```bash
python main.py sample_documents/sample.txt --output summary.txt
```

---

## 6. Direct Python Usage

The core algorithm can also be imported by another Python program:

```python
from document_summarizer import summarize_file

original_text, summary = summarize_file("sample_documents/sample.txt")

print(summary)
```

For raw text:

```python
from document_summarizer import summarize_text

text = "Your document text goes here..."
summary = summarize_text(text)

print(summary)
```

---

## 7. Optional Web Interface

The Flask interface is intentionally minimal. It demonstrates how the same summarization engine can be made accessible through a browser.

Start it:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Paste a document into the text box and click **Summarize**.

The Flask application does not contain the IR algorithm itself. It calls `summarize_text()` from `document_summarizer.py`.

This separation makes the project easier to understand and test.

---

## 8. Expected Output

For the included sample document:

```bash
python main.py sample_documents/sample.txt
```

The terminal displays a result similar to:

```text
======================================================================
DOCUMENT SUMMARIZATION RESULT
======================================================================
Input : sample_documents/sample.txt
Summary ratio : 35%

SUMMARY
----------------------------------------------------------------------
Information Retrieval is the process of finding useful information
from a large collection of data. TF-IDF is a classical technique
used to estimate how important a term is in a document. Extractive
summarization can use these weights to rank sentences.
======================================================================
```

Exact sentence selection can vary slightly with the number and wording of the input sentences.

---

## 9. Error Handling

The project intentionally includes basic but useful error handling.

### Empty document

```text
ERROR: The document is empty or contains no readable text.
```

### Missing file

```text
ERROR: File does not exist: example.txt
```

### Unsupported extension

For example:

```bash
python main.py data.docx
```

returns an error explaining that only `.txt` and `.pdf` are supported.

### Very short document

If a document has one or two sentences, the system returns the original text rather than producing an unhelpful empty summary.

### PDF without readable text

Some scanned/image-only PDFs do not contain an actual text layer. This prototype does not perform OCR, so such PDFs may return little or no text. OCR can be treated as future work.

---

## 10. Why Extractive Summarization?

This project uses extractive summarization because it is suitable for demonstrating IR fundamentals.

Advantages for a student project:

- Easy to understand
- No model training required
- No external API
- Works offline
- Uses classical IR concepts
- Easy to demonstrate during a viva
- Produces sentences directly from the original document

The limitation is that the summary may not be as grammatically smooth as an abstractive/generative model.

---

## 11. Main Functions

### `load_document()`

Identifies the file extension and sends the file to the appropriate reader.

### `read_text_file()`

Reads a UTF-8 text document.

### `read_pdf_file()`

Uses `pypdf` to extract text from PDF pages.

### `split_sentences()`

Performs lightweight sentence tokenization.

### `rank_sentences()`

Creates the TF-IDF matrix and calculates a relevance score for each sentence.

### `summarize_text()`

Selects approximately 35% of the highest-ranked sentences and restores their original order.

### `summarize_file()`

Connects file input to the summarization engine.

---

## 12. IR Concepts Demonstrated

| IR Concept | Implementation |
|---|---|
| Document processing | TXT/PDF loading |
| Tokenization | Rule-based sentence/token handling |
| Term weighting | TF-IDF |
| Stop-word removal | `TfidfVectorizer(stop_words="english")` |
| Vector-space model | Scikit-learn sparse TF-IDF matrix |
| Relevance scoring | Sum/average TF-IDF weights |
| Ranking | Descending sentence scores |
| Selection | Top ~35% sentences |
| Result ordering | Original sentence positions |

---

## 13. Limitations

This is intentionally a mini-project rather than a production summarization engine.

1. Sentence splitting is rule-based and may mishandle abbreviations.
2. PDF extraction may fail for scanned/image-only documents.
3. The algorithm is extractive, so it cannot generate new sentences.
4. TF-IDF alone does not understand deep semantic meaning.
5. The English stop-word list is used by scikit-learn.
6. No OCR is included.
7. No database or user authentication is required.
8. No external API is required.

---

## 14. Possible Future Enhancements

For a future version, students could add:

- NLTK or spaCy sentence tokenization
- OCR for scanned PDFs
- Cosine similarity with a document centroid
- TextRank summarization
- Multi-document summarization
- Hindi/Marathi language support
- Keyword extraction
- Summary evaluation using ROUGE
- File upload in the Flask interface
- Summary history using SQLite

These are deliberately outside the scope of the quick prototype.

---

## 15. Suggested Viva Explanation

**Question: Why did you use TF-IDF?**

Answer:

> TF-IDF is a classical Information Retrieval technique that estimates the importance of terms. In this project, each sentence is treated as a small document. Sentences containing important terms receive higher scores, so we can rank and select them for extractive summarization.

**Question: Why is this extractive?**

Answer:

> The system does not create new sentences. It selects important sentences from the original document and combines them into a shorter summary.

**Question: Why restore the original order?**

Answer:

> TF-IDF ranking gives importance order, but putting sentences back into their original order generally makes the resulting summary easier to read and preserves the source document's logical flow.

---

## 16. Academic Scope

This implementation is appropriate as a final-year undergraduate mini-project because it demonstrates the complete pipeline:

```text
Input Document
      ↓
Text Extraction
      ↓
Preprocessing
      ↓
Sentence Tokenization
      ↓
TF-IDF Representation
      ↓
Sentence Scoring
      ↓
Sentence Ranking
      ↓
Top 30–40% Selection
      ↓
Original Order Restoration
      ↓
Final Summary
```

It is small enough to understand, test, explain, and demonstrate during a project evaluation while still showing a genuine Information Retrieval technique.


## Web Upload Feature

The Flask interface now supports uploading **TXT, PDF, and DOCX** files directly from the browser. It also accepts pasted text and lets the user choose a 30%, 35%, or 40% summary ratio. PDF text is extracted with `pypdf`, while DOCX paragraphs and table cells are extracted with `python-docx`.

Run:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.
