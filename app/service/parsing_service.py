from pydantic import BaseModel
import fitz  # PyMuPDF

class ParsedPage(BaseModel):
    page_index: int
    text: str


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
            text = page.get_text("text")
            pages.append(ParsedPage(page_index=page_index, text=text))
    finally:
        doc.close()

    return ParsedPaper(
        paper_id=paper_id,
        pdf_path=pdf_path,
        num_pages=len(pages),
        pages=pages,
    )

