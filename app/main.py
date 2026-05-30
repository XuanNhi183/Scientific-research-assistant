from fastapi import FastAPI, UploadFile, HTTPException
import os
from app.schemas.upload import UploadResponse
from uuid import uuid4
import uvicorn

app = FastAPI()

@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )
    else:
        upload_location = "./data/uploads/"
        file_location = f"{upload_location}{file.filename}"
        
        os.makedirs(upload_location, exist_ok=True)
        
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        return UploadResponse(
            filename=file.filename,
            file_id=str(uuid4()),  
            file_path=file_location
        )

# http://127.0.0.1:8000/docs
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )