from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    paper_id: str | None = None
    
class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]    

class AnalyzeRequest(BaseModel):
    rawText: str
    title: str = "Tài liệu khoa học"
    