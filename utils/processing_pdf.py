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

    # Headings are usually short. If it's too long, it's likely a paragraph or citation.
    if len(text) < 3 or len(text) > 150:
        return False

    # If it ends with a period, it's likely a sentence (except for single words or short phrases)
    if text.endswith(".") and len(text.split()) > 4:
        return False

    # Check for at least some alphabet characters
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    # Arxiv headings typically have font_size strictly larger than body_size
    if font_size >= body_size + 0.5:
        return True

    # If font size is identical/very close, check if it matches a numbered heading pattern: "1 Introduction", "1.1 Background", "II. Method"
    import re
    if re.match(r"^([0-9]+\.?[0-9]*|[IVXLCDM]+\.?)\s+[A-Z]", text):
        return True

    return False