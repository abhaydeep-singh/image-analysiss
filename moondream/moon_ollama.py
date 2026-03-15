"""
To use in main.py, replace:
    from moondream.moon import analyze_image, build_pdf
with:
    from moondream.moon_ollama import analyze_image, build_pdf

Requirements:
    - Ollama installed and running (ollama serve)
    - Moondream pulled (ollama pull moondream)
    - pip install ollama
"""

import ollama
import base64
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)

PROMPT = """
Describe this image in detail. Cover:
1. Overall scene and environment
2. People present and what they are doing
3. Notable objects visible

Be factual, clear, and specific.
"""


# ── Image analysis ────────────────────────────────────────────────────────────
def analyze_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    try:
        response = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": PROMPT, "images": [image_data]}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Analysis failed: {str(e)}]"


# ── PDF generation ────────────────────────────────────────────────────────────
def build_pdf(results: list, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    page_w, page_h = landscape(A4)
    usable_w = page_w - 3 * cm
    usable_h = page_h - 4 * cm

    styles     = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )
    label_style = ParagraphStyle(
        "ImageLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    img_col_w = usable_w * 0.45
    txt_col_w = usable_w * 0.55
    story     = []

    for i, result in enumerate(results, 1):
        try:
            img_element = fit_image(result["path"], img_col_w - 1 * cm, usable_h - 1 * cm)
            img_cell    = [
                img_element,
                Spacer(1, 0.2 * cm),
                Paragraph(result["filename"], label_style),
            ]
        except Exception as e:
            img_cell = [Paragraph(f"[Could not load image: {e}]", body_style)]

        txt_cell = format_analysis(result["analysis"], body_style)

        layout_table = Table(
            [[img_cell, txt_cell]],
            colWidths=[img_col_w, txt_col_w],
            hAlign="LEFT"
        )
        layout_table.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("LINEAFTER",    (0, 0), (0, -1),  1, colors.HexColor("#dddddd")),
        ]))

        story.append(layout_table)
        if i < len(results):
            story.append(PageBreak())

    doc.build(story)


def fit_image(image_path: str, max_w: float, max_h: float) -> RLImage:
    with PILImage.open(image_path) as img:
        orig_w, orig_h = img.size
    ratio = min(max_w / orig_w, max_h / orig_h)
    return RLImage(image_path, width=orig_w * ratio, height=orig_h * ratio)


def format_analysis(text: str, body_style) -> list:
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            paragraphs.append(Spacer(1, 0.15 * cm))
            continue
        if line[0].isdigit() and "." in line[:3]:
            paragraphs.append(Paragraph(f"<b>{line}</b>", body_style))
        else:
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paragraphs.append(Paragraph(safe, body_style))
    return paragraphs