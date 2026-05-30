import re
from pydantic import BaseModel
import fitz  # PyMuPDF

class ParsedPage(BaseModel):
    page_index: int
    text: str
    raw_blocks: list = None
    layout_boxes: list = None


class ParsedPaper(BaseModel):
    paper_id: str
    pdf_path: str
    num_pages: int
    pages: list[ParsedPage]


def parse_pdf_to_pages(pdf_path: str, paper_id: str) -> ParsedPaper:
    doc = fitz.open(pdf_path)
    pages: list[ParsedPage] = []
    
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            
            # Text thuần (giữ nguyên để clean sau)
            plain_text = page.get_text("text")
            
            # Lưu blocks để detect heading chính xác
            raw_dict = page.get_text("dict")
            
            pages.append(ParsedPage(
                page_index=page_index, 
                text=plain_text,
                raw_blocks=raw_dict["blocks"]
            ))
    finally:
        doc.close()

    return ParsedPaper(
        paper_id=paper_id,
        pdf_path=pdf_path,
        num_pages=len(pages),
        pages=pages,
    )

def clean_page_texts(pages: list[ParsedPage]) -> list[ParsedPage]:
    # 1. Detect header/footer candidates
    first_lines = [p.text.splitlines()[0] if p.text.splitlines() else '' for p in pages]
    last_lines = [p.text.splitlines()[-1] if p.text.splitlines() else '' for p in pages]
    header_counts = {}
    footer_counts = {}
    for line in first_lines:
        header_counts[line] = header_counts.get(line, 0) + 1
    for line in last_lines:
        footer_counts[line] = footer_counts.get(line, 0) + 1
    header = max(header_counts, key=header_counts.get) if header_counts else ''
    footer = max(footer_counts, key=footer_counts.get) if footer_counts else ''
    header_thresh = len(pages) // 2
    footer_thresh = len(pages) // 2

    cleaned_pages = []
    for p in pages:
        lines = p.text.splitlines()
        # Remove header/footer if repeated
        if lines and header and lines[0] == header and header_counts[header] > header_thresh:
            lines = lines[1:]
        if lines and footer and lines[-1] == footer and footer_counts[footer] > footer_thresh:
            lines = lines[:-1]
        # Hyphenation fix & join lines
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.rstrip().endswith('-') and i+1 < len(lines):
                next_line = lines[i+1].lstrip()
                if next_line and next_line[0].islower():
                    # Remove hyphen, join
                    line = line.rstrip()[:-1] + next_line
                    i += 1
            new_lines.append(line)
            i += 1
        # Join lines, fix linebreaks in middle of sentence
        text = '\n'.join(new_lines)
        text = re.sub(r'(?<![.!?:])\n([a-z])', r' \1', text)  # join lines not ending with .!?:
        # Normalize whitespace
        text = re.sub(r'(?<![.!?:])\n([a-z])', r' \1', text)
        text = re.sub(r'\s{2,}', ' ', text)
        cleaned_pages.append(ParsedPage(page_index=p.page_index, text=text.strip()))
    return cleaned_pages


COMMON_SECTIONS = {
    "ABSTRACT", "INTRODUCTION", "RELATED WORK", "RELATED WORKS", 
    "MEMORY OPTIMIZATION", "TRADE COMPUTATION FOR MEMORY",
    "EXPERIMENTS", "EXPERIMENT", "RESULTS", "DISCUSSION", 
    "CONCLUSION", "CONCLUSIONS", "ACKNOWLEDGMENTS", 
    "ACKNOWLEDGMENT", "REFERENCES", "APPENDIX"
}

def detect_sections(pages: list[ParsedPage]) -> list[dict]:
    segments = []
    current_section = "Title and Authors"
    buffer = []

    print("=== DEBUG MODE ===\n")

    for page in pages:
        if not page.raw_blocks:
            continue

        page_segments, new_current = _detect_page_debug(page, current_section)
        segments.extend(page_segments)
        current_section = new_current

    print(f"\n✅ Hoàn thành! Tổng segments: {len(segments)}\n")
    return segments


def _detect_page_debug(page: ParsedPage, current_section: str):
    segments = []
    buffer = []
    current = current_section

    for block_idx, block in enumerate(page.raw_blocks):
        for line_idx, line in enumerate(block.get("lines", [])):
            line_text = ""
            max_size = 0.0
            is_bold = False

            for span in line.get("spans", []):
                line_text += span["text"]
                max_size = max(max_size, round(span["size"], 1))
                if (span["flags"] & 16) or ("bold" in span["font"].lower()) or max_size >= 11.0:
                    is_bold = True

            stripped = line_text.strip()
            if not stripped or len(stripped) < 5:
                continue

            is_heading = _is_heading_debug(stripped, max_size, is_bold)

            # DEBUG
            if max_size >= 10.5 or is_bold or re.match(r'^\d', stripped):
                print(f"Page {page.page_index:2d} | Size={max_size:4.1f} | Bold={is_bold} | Heading={is_heading} | {stripped[:85]}")

            if is_heading:
                if buffer:
                    segments.append({
                        "section": current,
                        "page_index": page.page_index,
                        "text": " ".join(buffer).strip()
                    })
                    buffer = []
                current = stripped
            else:
                buffer.append(stripped)

    # Đoạn cuối trang
    if buffer:
        segments.append({
            "section": current,
            "page_index": page.page_index,
            "text": " ".join(buffer).strip()
        })

    return segments, current


def _is_heading_debug(text: str, font_size: float, is_bold: bool) -> bool:
    text = text.strip()
    if len(text) > 170 or len(text) < 5:
        return False

    # Rule rộng
    if font_size >= 10.8:          # Giảm thấp để bắt
        return True

    if re.match(r'^\s*\d+(\.\d+)*', text):   # Bắt mọi số thứ tự
        return True

    # Common sections
    text_clean = re.sub(r'[^A-Z ]', '', text.upper())
    for sec in COMMON_SECTIONS:
        if sec.replace(" ", "") in text_clean:
            return True

    if is_bold and len(text.split()) <= 18:
        return True

    return False