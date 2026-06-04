from service.document_service import doc_service
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schemas.chunk import Chunk, ChunkMetadata
from uuid import uuid4
import tiktoken
from dataclasses import dataclass
from utils.processing_pdf import extract_lines, get_body_font_size, is_heading

encoding = tiktoken.encoding_for_model(
    "text-embedding-3-small"
)

@dataclass
class SectionInfo:
    title: str
    text: str
    page_start: int
    page_end: int


def extract_sections(pdf_path: str):
    lines = extract_lines(pdf_path)

    if not lines:
        return []

    body_size = get_body_font_size(lines)

    headings = []

    for idx, line in enumerate(lines):
        if is_heading(
            line["text"],
            line["font_size"],
            body_size,
        ):
            headings.append(
                {
                    "index": idx,
                    "title": line["text"],
                    "page": line["page"],
                }
            )

    if not headings:
        return []

    # bỏ mọi thứ trước ABSTRACT
    abstract_idx = None

    for i, h in enumerate(headings):
        if "ABSTRACT" in h["title"].upper():
            abstract_idx = i
            break

    if abstract_idx is None:
        return []

    headings = headings[abstract_idx:]

    sections = []

    for i in range(len(headings)):
        current = headings[i]

        start_idx = current["index"]

        if i < len(headings) - 1:
            end_idx = headings[i + 1]["index"]
        else:
            end_idx = len(lines)

        section_lines = [
            line["text"]
            for line in lines[start_idx:end_idx]
        ]

        section_text = "\n".join(section_lines)

        page_start = current["page"]

        if i < len(headings) - 1:
            page_end = headings[i + 1]["page"]
        else:
            page_end = lines[-1]["page"]

        sections.append(
            SectionInfo(
                title=current["title"],
                text=section_text,
                page_start=page_start,
                page_end=page_end,
                
            )
        )

    return sections

def extract_document(file_id: str):
    pages = doc_service.extract_pages(file_id)
    full_text = ""
    page_map = []
    current_position = 0
    for page in pages:
        page_text = page["text"]
        start = current_position
        # Keep page boundaries in the concatenated text for stable offsets.
        full_text += page_text + " "
        current_position += len(page_text) + 2

        page_map.append({
            "page": page["page_number"],
            "char_start": start,
            "char_end": current_position,
        })

    return {
        "full_text": full_text,
        "page_map": page_map,
    }


def chunk_sections(sections, chunk_size, overlap,):
    results = []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n","\n",". "," ",""]
    )

    for section in sections:
        docs = splitter.create_documents(
            [section.text]
        )

        for doc in docs:
            results.append(
                {
                    "doc": doc,
                    "section": section,
                }
            )

    return results


def find_page_for_chunk(char_start: int, char_end: int, page_map: list[dict]):
    page_start = None
    page_end = None
    
    for page in page_map:
        if (
            page["char_start"] <= char_start < page["char_end"]
        ):
            page_start = page["page"]
            
        if (
            page["char_start"] <= char_end < page["char_end"]
        ):
            page_end = page["page"]
        if page_start is not None and page_end is not None:
            break
    return page_start, page_end


def get_chunks(file_id: str, chunk_size: int, overlap: int):
    doc_info = doc_service.get_document(file_id)
    results = []
    chunk_index = 0

    document = extract_document(file_id)
    page_map = document["page_map"]

    sections = extract_sections(doc_info["file_path"])
    documents = chunk_sections(sections, chunk_size, overlap)

    for item in documents:
        doc = item["doc"]
        section = item["section"]
        text = doc.page_content
        char_start = doc.metadata["start_index"]
        if char_start < 0:
            continue
        char_end = char_start + len(text) - 1
        page_start, page_end = find_page_for_chunk(char_start, char_end, page_map)
        
        if page_end is None:
            page_end = page_start

        metadata = ChunkMetadata(
            paper_id=file_id,
            title=doc_info["filename"],
            section=section.title,
            page_start=page_start,
            page_end=page_end,
            char_start=char_start,
            char_end=char_end,
            source_pdf=doc_info["file_path"],
        )
        results.append(
            Chunk(
                chunk_id=str(uuid4()),
                chunk_index=chunk_index,
                text=text,
                word_count=len(text.split()),
                token_count=len(encoding.encode(text)),
                metadata=metadata,
            )
        )
        chunk_index += 1

    return results