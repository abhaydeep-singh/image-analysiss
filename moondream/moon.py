IMAGE_FOLDER = "./assets"      # folder with your images
OUTPUT_PDF   = "./output/report_olama.pdf"

import ollama
import base64
import argparse
import sys
from pathlib import Path
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

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


def get_image_files(folder: Path) -> list[Path]:
    return [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ── PDF generation ────────────────────────────────────────────────────────────
def build_pdf(results: list[dict], output_path: str):
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

    styles = getSampleStyleSheet()

    image_label_style = ParagraphStyle(
        "ImageLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )

    img_col_w = usable_w * 0.45
    txt_col_w = usable_w * 0.55
    usable_h = page_h - 4 * cm  # account for top + bottom margins
    story = []

    for i, result in enumerate(results, 1):
        # Image cell
        try:
            img_element = fit_image(result["path"], img_col_w - 1 * cm, usable_h - 1 * cm)
            img_cell = [
                img_element,
                Spacer(1, 0.2 * cm),
                Paragraph(result["filename"], image_label_style),
            ]
        except Exception as e:
            img_cell = [Paragraph(f"[Could not load image: {e}]", body_style)]

        # Analysis cell
        txt_cell = format_analysis(result["analysis"], body_style)

        layout_table = Table(
            [[img_cell, txt_cell]],
            colWidths=[img_col_w, txt_col_w],
            hAlign="LEFT"
        )
        layout_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEAFTER", (0, 0), (0, -1), 1, colors.HexColor("#dddddd")),
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=IMAGE_FOLDER)
    parser.add_argument("--output", type=str, default=OUTPUT_PDF)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    image_files = get_image_files(folder)
    if not image_files:
        print(f"❌ No images found in: {folder}")
        sys.exit(1)

    print(f"📂 Found {len(image_files)} image(s)")
    print(f"🤖 Analyzing with Moondream...\n")

    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {img_path.name} ...", end=" ", flush=True)
        analysis = analyze_image(str(img_path))
        print("done")
        results.append({
            "filename": img_path.name,
            "path": str(img_path),
            "analysis": analysis,
        })

    print(f"\n📄 Generating PDF...")
    build_pdf(results, args.output)
    print(f"✅ Saved to: {args.output}")


if __name__ == "__main__":
    main()
