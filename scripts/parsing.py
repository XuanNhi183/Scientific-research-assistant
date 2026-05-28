from app.service.parsing_service import parse_pdf_to_pages

if __name__ == "__main__":
    pdf_path = "data/raw_papers/paper_001.pdf"
    paper_id = "paper_001"
    parsed_paper = parse_pdf_to_pages(pdf_path, paper_id)
    print(f"Parsed paper: {parsed_paper.paper_id}, num pages: {parsed_paper.num_pages}")
    for page in parsed_paper.pages:
        print(f"Page {page.page_index}: {len(page.text)} characters")