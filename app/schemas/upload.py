from pydantic import BaseModel
 
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_path: str
    