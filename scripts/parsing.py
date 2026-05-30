
from app.service.parsing_service import parse_pdf_to_pages, clean_page_texts, detect_sections

if __name__ == "__main__":
    pdf_path = "data/raw_papers/paper_001.pdf"
    paper_id = "paper_001"
    parsed_paper = parse_pdf_to_pages(pdf_path, paper_id)
    cleaning_result = clean_page_texts(parsed_paper.pages)
    sectioned_result = detect_sections(cleaning_result)
    print(f"Parsed paper: {parsed_paper.paper_id}, num pages: {parsed_paper.num_pages}")
        
    print("\nSectioned result:")
    for segment in sectioned_result:
        print(f"Section: {segment['section']}, Page: {segment['page_index']}, Characters: {len(segment['text'])}")
    for sec in sectioned_result[:15]:
        print(f"Page {sec['page_index']:2d} | {sec['section'][:80]:80} | {len(sec['text'])} chars")