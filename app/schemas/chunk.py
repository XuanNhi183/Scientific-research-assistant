from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str | None = None
    page: int | None = None
    chunk_index: int
    
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_pdf: str | None = None