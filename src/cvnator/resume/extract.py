"""Raw text extraction from uploaded CV files.

Deliberately dumb and deterministic: no model is involved. Getting clean text out of
the file is a parsing problem, not a reasoning problem, and keeping it separate means
a bad extraction is visible as a bad extraction rather than as a hallucinating parser.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedDocument(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Bullet glyphs confuse downstream tokenising; normalise to a hyphen.
    text = re.sub(r"^[\s]*[•▪◦‣·–—*]\s*", "- ", text, flags=re.MULTILINE)
    return text.strip()


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pragma: no cover - malformed page, keep the rest
            pages.append("")
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    blocks: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def extract_text(path: Path) -> str:
    """Return normalized plain text from a PDF, DOCX, TXT or MD file."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(
            f"Unsupported CV format {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".pdf":
        raw = _extract_pdf(path)
    elif suffix == ".docx":
        raw = _extract_docx(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    text = _normalize(raw)
    if len(text) < 40:
        raise UnsupportedDocument(
            f"Extracted only {len(text)} characters from {path.name}. "
            "The file may be a scanned image; CVnator does not OCR CVs. "
            "Please upload a text-based PDF or a DOCX."
        )
    return text
