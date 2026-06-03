import fitz
from collections import Counter


def extract_lines(pdf_path: str):
    doc = fitz.open(pdf_path)

    lines = []

    for page_num, page in enumerate(doc, start=1):
        data = page.get_text("dict")

        for block in data["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = "".join(
                    span["text"]
                    for span in line["spans"]
                ).strip()

                if not text:
                    continue

                sizes = [
                    span["size"]
                    for span in line["spans"]
                    if span["text"].strip()
                ]

                avg_size = sum(sizes) / len(sizes)

                lines.append({
                    "page": page_num,
                    "text": text,
                    "font_size": avg_size,
                })

    return lines

def get_body_font_size(lines):
    rounded_sizes = [
        round(line["font_size"], 1)
        for line in lines
    ]

    counter = Counter(rounded_sizes)

    return counter.most_common(1)[0][0]


def is_heading(text: str, font_size: float, body_size: float,):
    text = text.strip()

    if len(text) < 3:
        return False

    if ":" in text:
        return False

    if font_size < body_size + 0.5:
        return False

    alpha_chars = [
        c
        for c in text if c.isalpha()
    ]

    if not alpha_chars:
        return False

    upper_ratio = (
        sum(c.isupper() for c in alpha_chars)
        / len(alpha_chars)
    )

    return upper_ratio > 0.8