from pydantic import BaseModel
 
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_path: str
    

class DocumentContentResponse(BaseModel):
    file_id: str
    content: str