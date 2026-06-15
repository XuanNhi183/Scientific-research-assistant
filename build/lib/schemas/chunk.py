from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    paper_id: str
    title: str

    section: str | None = None
    category: str | None = None

    page_start: int | None = None
    page_end: int | None = None

    char_start: int | None = None
    char_end: int | None = None

    source_pdf: str | None = None
    
    
class Chunk(BaseModel):
    chunk_id: str
    chunk_index: int

    text: str

    word_count: int
    token_count: int | None = None

    embedding: list[float] | None = None

    metadata: ChunkMetadata