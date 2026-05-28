from pydantic import BaseModel
from datetime import datetime

class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None = None
    source: str | None = None
    pdf_path: str
    
    abstract: str | None = None
    doi: str | None = None
    venue: str | None = None
    keywords: list[str] | None = None
    num_pages: int | None = None
    checksum: str | None = None
    created_at: datetime | None = None
    