from fastapi import FastAPI, UploadFile, HTTPException
from app.schemas.document import UploadResponse, DocumentContentResponse
import uvicorn
from app.service.document_service import doc_service

app = FastAPI()

@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )
    else:
       result = await doc_service.upload_file(file)
       return UploadResponse(**result)


@app.get("/document/{file_id}/content", response_model=DocumentContentResponse)
async def get_document_content(file_id: str):
    content = doc_service.extract_text(file_id)
    return DocumentContentResponse(file_id=file_id, content=content)





# http://127.0.0.1:8000/docs
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )