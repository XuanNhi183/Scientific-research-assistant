from service.document_service import doc_service
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from schemas.chunk import Chunk, ChunkMetadata
from uuid import uuid4
import tiktoken
from dataclasses import dataclass
from utils.processing_pdf import extract_lines, get_body_font_size, is_heading

encoding = tiktoken.encoding_for_model(
    "text-embedding-3-small"
)


def is_data_chunk(text: str) -> bool:
    """
    Detect chunks that are primarily data tables, frequency lists, or raw statistics
    with no narrative content — these produce no valid QA pairs and should be skipped.

    Uses three independent signals. A chunk is flagged if 2 or more signals fire
    (lenient consensus to avoid dropping legitimate result-discussion chunks).

    Signals
    -------
    1. numeric_line_ratio  : fraction of lines where >= 50% of tokens are numeric/punctuation.
    2. avg_line_length     : mean character count per non-empty line.
    3. unique_token_ratio  : unique_tokens / total_tokens.
    4. min_char_length     : absolute minimum guard (checked first, before signal scoring).
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True

    # Absolute minimum length guard
    if len(text.strip()) < 150:
        return True

    # Signal 1: numeric line ratio
    numeric_line_count = 0
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        numeric_tokens = sum(
            1 for t in tokens
            if t.replace(",", "").replace(".", "").replace("%", "").replace("-", "").isdigit()
        )
        if numeric_tokens / len(tokens) >= 0.5:
            numeric_line_count += 1
    numeric_ratio = numeric_line_count / len(lines)

    # Signal 2: average line length
    avg_line_len = sum(len(l) for l in lines) / len(lines)

    # Signal 3: unique token ratio
    all_tokens = text.lower().split()
    unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

    # Signal 4: text density
    letters = sum(1 for c in text if c.isalpha())
    letter_ratio = letters / max(1, len(text))

    if letter_ratio < 0.60:
        return True

    flags = [
        numeric_ratio >= 0.35,  # relaxed from 0.40
        avg_line_len <= 50,     # relaxed from 40
        unique_ratio <= 0.40,   # relaxed from 0.35
    ]

    # Require 2 of 3 signals — avoids false positives on result discussion sections
    return sum(flags) >= 2

def init_marker_models():
    models = load_all_models()
    return models

def extract_with_marker(pdf_path:str, model_lst) -> str:
    full_text, images, out_meta = convert_single_pdf(pdf_path, model_lst)
    return full_text

def chunk_markdown(markdown_content:str)-> list[Chunk]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_content)
    chunks = []
    for i, doc in enumerate(md_header_splits):
        text = doc.page_content
        if not text.strip():
            continue
        langchain_meta = doc.metadata
        section_title = langchain_meta.get("Header 3") or langchain_meta.get("Header 2") or langchain_meta.get("Header 1") or "Unknown"
        
        metadata = ChunkMetadata(
            paper_id="",
            title="",
            section=section_title,
            page_start=None,  
            page_end=None,            
        )
        chunks.append(
            Chunk(
                chunk_id=str(uuid4()),
                chunk_index=i,
                text=text,
                word_count=len(text.split()),
                token_count=len(encoding.encode(text)),
                metadata=metadata,
            )
        )
    return chunks
    

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

    # Bỏ mọi thứ trước ABSTRACT hoặc INTRODUCTION
    abstract_idx = None

    for i, h in enumerate(headings):
        if "ABSTRACT" in h["title"].upper() or "INTRODUCTION" in h["title"].upper():
            abstract_idx = i
            break

    # Nếu không tìm thấy, bắt đầu từ heading đầu tiên thay vì bỏ qua toàn bộ document
    if abstract_idx is None:
        abstract_idx = 0

    valid_headings = headings[abstract_idx:]

    sections = []
    
    # 1. Capture Metadata (Title, Authors, Affiliations) before the Abstract
    metadata_end_idx = valid_headings[0]["index"] if valid_headings else len(lines)
    if metadata_end_idx > 0:
        metadata_lines = [line["text"] for line in lines[0:metadata_end_idx]]
        if metadata_lines:
            metadata_text = "Here are the authors and affiliations of this paper:\n\n" + "\n".join(metadata_lines)
            sections.append(
                SectionInfo(
                    title="TITLE & AUTHORS",
                    text=metadata_text,
                    page_start=lines[0]["page"],
                    page_end=lines[metadata_end_idx - 1]["page"],
                )
            )

    # 2. Capture the rest of the headings
    for i in range(len(valid_headings)):
        current = valid_headings[i]
        start_idx = current["index"]

        if i < len(valid_headings) - 1:
            end_idx = valid_headings[i + 1]["index"]
        else:
            end_idx = len(lines)

        section_lines = [
            line["text"]
            for line in lines[start_idx:end_idx]
        ]

        section_text = "\n".join(section_lines)

        page_start = current["page"]

        if i < len(valid_headings) - 1:
            page_end = valid_headings[i + 1]["page"]
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


def chunk_sections(sections, chunk_size, overlap):
    results = []
    dropped_data = 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for section in sections:
        docs = splitter.create_documents([section.text])

        for doc in docs:
            if is_data_chunk(doc.page_content):
                dropped_data += 1
                continue
            results.append({"doc": doc, "section": section})

    if dropped_data:
        print(f"  [chunking] dropped {dropped_data} data/table chunks (no narrative content)")

    return results


def find_page_for_chunk(char_start: int, char_end: int, page_map: list[dict]):
    page_start = None
    page_end = None

    for page in page_map:
        if page["char_start"] <= char_start < page["char_end"]:
            page_start = page["page"]
        if page["char_start"] <= char_end < page["char_end"]:
            page_end = page["page"]
        if page_start is not None and page_end is not None:
            break

    return page_start, page_end


def get_chunks(file_id: str, chunk_size: int, overlap: int):
    doc_info = doc_service.get_document(file_id)
    results = []
    chunk_index = 0

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

        page_start = section.page_start
        page_end = section.page_end

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