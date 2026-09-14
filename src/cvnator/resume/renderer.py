"""ATS-friendly resume rendering to DOCX and PDF.

Template design follows what resume parsers can actually read:

* single column, no tables, no text boxes, no headers/footers, no graphics
* real selectable text only - never an image containing text
* standard, predictable section headings in the document's language
* contact details as plain text lines, not icons
* dates as text next to the role, not in a separate visual column

The result looks plain. That is the point: a parser that mis-reads a two-column
layout will drop half a career, and no amount of keyword tuning recovers from that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cvnator.observability import get_logger
from cvnator.resume.optimizer import ResumeDocument

log = get_logger(__name__)

SECTION_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "summary": "PROFESSIONAL SUMMARY",
        "experience": "PROFESSIONAL EXPERIENCE",
        "skills": "SKILLS",
        "projects": "PROJECTS",
        "education": "EDUCATION",
        "certifications": "CERTIFICATIONS",
        "languages": "LANGUAGES",
        "present": "Present",
        "technologies": "Technologies",
    },
    "pt": {
        "summary": "RESUMO PROFISSIONAL",
        "experience": "EXPERIÊNCIA PROFISSIONAL",
        "skills": "COMPETÊNCIAS",
        "projects": "PROJETOS",
        "education": "FORMAÇÃO ACADÊMICA",
        "certifications": "CERTIFICAÇÕES",
        "languages": "IDIOMAS",
        "present": "Atual",
        "technologies": "Tecnologias",
    },
    "es": {
        "summary": "RESUMEN PROFESIONAL",
        "experience": "EXPERIENCIA PROFESIONAL",
        "skills": "COMPETENCIAS",
        "projects": "PROYECTOS",
        "education": "FORMACIÓN ACADÉMICA",
        "certifications": "CERTIFICACIONES",
        "languages": "IDIOMAS",
        "present": "Actual",
        "technologies": "Tecnologías",
    },
}


def labels_for(language: str) -> dict[str, str]:
    return SECTION_LABELS.get((language or "en").lower()[:2], SECTION_LABELS["en"])


@dataclass(slots=True)
class RenderedFile:
    path: Path
    sha256: str
    size_bytes: int
    template: str


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _period(start: str, end: str, is_current: bool, present_label: str) -> str:
    tail = present_label if is_current or not end else end
    return " - ".join(p for p in (start, tail) if p)


def render_docx(
    document: ResumeDocument, output_path: Path, *, language: str = "en"
) -> RenderedFile:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    labels = labels_for(language)
    doc = docx.Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    def heading(text: str) -> None:
        para = doc.add_paragraph()
        para.paragraph_format.space_before = Pt(10)
        para.paragraph_format.space_after = Pt(2)
        run = para.add_run(text)
        run.bold = True
        run.font.size = Pt(11.5)

    def body(text: str, *, bullet: bool = False, bold: bool = False) -> None:
        para = doc.add_paragraph(style="List Bullet" if bullet else None)
        para.paragraph_format.space_after = Pt(1)
        run = para.add_run(text)
        run.bold = bold

    name_para = doc.add_paragraph()
    name_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    name_run = name_para.add_run(document.full_name)
    name_run.bold = True
    name_run.font.size = Pt(18)

    if document.headline:
        body(document.headline)
    if document.contact_lines:
        body(" | ".join(document.contact_lines))

    if document.professional_summary:
        heading(labels["summary"])
        body(document.professional_summary)

    if document.experiences:
        heading(labels["experience"])
        for exp in document.experiences:
            body(f"{exp.job_title} — {exp.company}", bold=True)
            meta = [_period(exp.start_date, exp.end_date, exp.is_current, labels["present"])]
            if exp.location:
                meta.append(exp.location)
            body(" | ".join(m for m in meta if m))
            for bullet in exp.bullets:
                body(bullet, bullet=True)
            if exp.technologies:
                body(f"{labels['technologies']}: {', '.join(exp.technologies)}")

    if document.skills:
        heading(labels["skills"])
        body(", ".join(document.skills))

    if document.projects:
        heading(labels["projects"])
        for proj in document.projects:
            body(proj.name, bold=True)
            if proj.description:
                body(proj.description)
            if proj.technologies:
                body(f"{labels['technologies']}: {', '.join(proj.technologies)}")
            if proj.url:
                body(proj.url)

    for key, items in (
        ("education", document.education),
        ("certifications", document.certifications),
    ):
        if items:
            heading(labels[key])
            for item in items:
                body(item, bullet=True)

    if document.languages:
        heading(labels["languages"])
        body(", ".join(document.languages))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    return RenderedFile(
        path=output_path,
        sha256=_sha256_of(output_path),
        size_bytes=output_path.stat().st_size,
        template="ats_plain_docx",
    )


def render_pdf(
    document: ResumeDocument, output_path: Path, *, language: str = "en"
) -> RenderedFile:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Flowable,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    labels = labels_for(language)
    base = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "CVName", parent=base["Title"], fontSize=18, leading=21, alignment=TA_LEFT,
        spaceAfter=2, fontName="Helvetica-Bold",
    )
    heading_style = ParagraphStyle(
        "CVHeading", parent=base["Heading2"], fontSize=11.5, leading=13,
        spaceBefore=10, spaceAfter=3, fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "CVBody", parent=base["BodyText"], fontSize=10, leading=13, spaceAfter=2,
    )
    role_style = ParagraphStyle(
        "CVRole", parent=body_style, fontName="Helvetica-Bold", spaceBefore=6,
    )
    meta_style = ParagraphStyle("CVMeta", parent=body_style, textColor=HexColor("#444444"))

    def esc(text: str) -> str:
        return (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    story: list[Flowable] = [Paragraph(esc(document.full_name), name_style)]
    if document.headline:
        story.append(Paragraph(esc(document.headline), body_style))
    if document.contact_lines:
        story.append(Paragraph(esc(" | ".join(document.contact_lines)), meta_style))

    def section(title: str) -> None:
        story.append(Paragraph(esc(title), heading_style))

    def bullets(items: list[str]) -> None:
        if not items:
            return
        # types-reportlab does not model ListItem as an accepted ListFlowable child,
        # though it is the documented (and only) way to build a bulleted list.
        entries = [ListItem(Paragraph(esc(i), body_style), leftIndent=10) for i in items]
        story.append(
            ListFlowable(entries, bulletType="bullet", start="-", leftIndent=12)  # type: ignore[arg-type]
        )

    if document.professional_summary:
        section(labels["summary"])
        story.append(Paragraph(esc(document.professional_summary), body_style))

    if document.experiences:
        section(labels["experience"])
        for exp in document.experiences:
            story.append(Paragraph(esc(f"{exp.job_title} — {exp.company}"), role_style))
            meta = [_period(exp.start_date, exp.end_date, exp.is_current, labels["present"])]
            if exp.location:
                meta.append(exp.location)
            story.append(Paragraph(esc(" | ".join(m for m in meta if m)), meta_style))
            bullets(exp.bullets)
            if exp.technologies:
                story.append(
                    Paragraph(
                        esc(f"{labels['technologies']}: {', '.join(exp.technologies)}"),
                        meta_style,
                    )
                )

    if document.skills:
        section(labels["skills"])
        story.append(Paragraph(esc(", ".join(document.skills)), body_style))

    if document.projects:
        section(labels["projects"])
        for proj in document.projects:
            story.append(Paragraph(esc(proj.name), role_style))
            if proj.description:
                story.append(Paragraph(esc(proj.description), body_style))
            if proj.technologies:
                story.append(
                    Paragraph(
                        esc(f"{labels['technologies']}: {', '.join(proj.technologies)}"),
                        meta_style,
                    )
                )
            if proj.url:
                story.append(Paragraph(esc(proj.url), meta_style))

    for key, items in (
        ("education", document.education),
        ("certifications", document.certifications),
    ):
        if items:
            section(labels[key])
            bullets(items)

    if document.languages:
        section(labels["languages"])
        story.append(Paragraph(esc(", ".join(document.languages)), body_style))

    story.append(Spacer(1, 4 * mm))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"{document.full_name} - CV",
        author=document.full_name,
    )
    pdf.build(story)
    return RenderedFile(
        path=output_path,
        sha256=_sha256_of(output_path),
        size_bytes=output_path.stat().st_size,
        template="ats_plain_pdf",
    )
