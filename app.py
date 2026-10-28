"""Flask web interface for the IR Document Summarization System.

Features:
- Paste text directly into the browser.
- Upload TXT, PDF, or DOCX documents.
- Select a summary ratio between 30% and 40%.
- Reuses the same TF-IDF summarization core used by the CLI.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

from pathlib import Path
import tempfile

from flask import Flask, render_template_string, request

from document_summarizer import (
    SUPPORTED_EXTENSIONS,
    SummarizationError,
    summarize_file,
    summarize_text,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Summarization System</title>
<style>
    * { box-sizing: border-box; }
    body { margin:0; font-family:Arial, sans-serif; background:#f5f7fb; color:#172033; }
    .wrap { max-width:1050px; margin:40px auto; padding:0 20px; }
    .card { background:white; border-radius:16px; padding:28px; box-shadow:0 8px 30px rgba(20,35,60,.08); margin-bottom:22px; }
    h1 { margin-top:0; color:#0d1f40; }
    h2 { color:#0d1f40; }
    .sub { color:#667085; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
    textarea { width:100%; min-height:300px; padding:14px; border:1px solid #d0d5dd; border-radius:10px; resize:vertical; font-size:15px; }
    input[type=file], select { width:100%; padding:12px; border:1px solid #d0d5dd; border-radius:10px; background:white; }
    button { margin-top:16px; padding:12px 22px; border:0; border-radius:9px; background:#1a4fcc; color:white; font-weight:700; cursor:pointer; }
    button:hover { opacity:.92; }
    .result { white-space:pre-wrap; line-height:1.7; background:#f8fafc; border-left:4px solid #1a4fcc; padding:18px; border-radius:8px; }
    .error { color:#b42318; background:#fef3f2; padding:12px; border-radius:8px; }
    .meta { color:#667085; font-size:14px; margin-top:8px; }
    .hint { font-size:13px; color:#667085; margin-top:7px; }
    @media(max-width:750px){ .grid{grid-template-columns:1fr;} .wrap{margin:20px auto;} }
</style>
</head>
<body>
<div class="wrap">
    <div class="card">
        <h1>IR Document Summarization System</h1>
        <p class="sub">Extractive summarization using TF-IDF. Upload a TXT, PDF, or DOCX document, or paste text directly.</p>

        <form method="post" enctype="multipart/form-data">
            <div class="grid">
                <div>
                    <h2>Upload Document</h2>
                    <input type="file" name="document" accept=".txt,.pdf,.docx">
                    <p class="hint">Supported formats: TXT, PDF, DOCX • Maximum size: 20 MB</p>
                </div>
                <div>
                    <h2>Summary Ratio</h2>
                    <select name="ratio">
                        <option value="0.30" {% if ratio == 0.30 %}selected{% endif %}>30%</option>
                        <option value="0.35" {% if ratio == 0.35 %}selected{% endif %}>35% (Recommended)</option>
                        <option value="0.40" {% if ratio == 0.40 %}selected{% endif %}>40%</option>
                    </select>
                </div>
            </div>

            <h2>Or Paste Text</h2>
            <textarea name="text" placeholder="Paste your document text here...">{{ text }}</textarea>
            <button type="submit">Generate Summary</button>
        </form>
    </div>

    {% if error %}
    <div class="card error"><strong>Error:</strong> {{ error }}</div>
    {% endif %}

    {% if summary %}
    <div class="card">
        <h2>Generated Summary</h2>
        <p class="meta">Source: {{ source_name }} | Summary ratio: {{ ratio_percent }}%</p>
        <div class="result">{{ summary }}</div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""


def allowed_upload(filename: str) -> bool:
    return bool(filename) and Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = ""
    error = ""
    source = "Pasted text"
    ratio = 0.35

    if request.method == "POST":
        try:
            ratio = float(request.form.get("ratio", "0.35"))
            text = request.form.get("text", "")
            uploaded = request.files.get("document")

            if uploaded and uploaded.filename:
                if not allowed_upload(uploaded.filename):
                    raise SummarizationError(
                        "Unsupported upload. Please choose a TXT, PDF, or DOCX file."
                    )

                suffix = Path(uploaded.filename).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    uploaded.save(tmp.name)
                    temp_path = Path(tmp.name)

                try:
                    _original, summary = summarize_file(str(temp_path), ratio=ratio)
                    source = uploaded.filename
                finally:
                    temp_path.unlink(missing_ok=True)
            elif text.strip():
                summary = summarize_text(text, ratio=ratio)
            else:
                raise SummarizationError("Upload a TXT/PDF/DOCX file or paste text first.")

        except (SummarizationError, ValueError) as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        text=text,
        summary=summary,
        error=error,
        source_name=source,
        ratio=ratio,
        ratio_percent=int(ratio * 100),
    )


@app.errorhandler(413)
def too_large(_error):
    return render_template_string(
        HTML,
        text="",
        summary="",
        error="The uploaded file is larger than the 20 MB limit.",
        source_name="",
        ratio=0.35,
        ratio_percent=35,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
