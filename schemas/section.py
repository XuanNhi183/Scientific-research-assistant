from pydantic import BaseModel

class Section(BaseModel):
    title: str
    text: str

    page_start: int | None = None
    page_end: int | None = None
    
    char_start: int | None = None
    char_end: int | None = None