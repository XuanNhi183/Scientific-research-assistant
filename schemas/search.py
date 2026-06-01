from pydantic import BaseModel

class SearchRequest(BaseModel):
    question: str