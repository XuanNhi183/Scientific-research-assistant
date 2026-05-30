from fastapi import UploadFile, HTTPException
import os
from uuid import uuid4
import json
import fitz

class DocumentService:
    async def upload_file(self, file: UploadFile):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are allowed."
                )
            
        upload_location = "./data/uploads/"
        os.makedirs(upload_location, exist_ok=True)
        
        file_id = str(uuid4())
        file_location = f"{upload_location}{file_id}.pdf"
        
        with open(file_location, "wb") as f:
            f.write(await file.read())
            
        self.save_metadata(file_id=file_id, filename=file.filename, file_path=file_location)
        
        return {
        "filename": file.filename,
        "file_id": file_id,
        "file_path": file_location,
    }
    
    def save_metadata(self,file_id: str,filename: str,file_path: str):
        metadata_location = "./data/metadata.jsonl"
        os.makedirs(os.path.dirname(metadata_location), exist_ok=True)
        record = {
                    "file_id": file_id,
                    "filename": filename,
                    "file_path": file_path
                }
        with open(metadata_location, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        return record
        
        
    def get_document(self, file_id:str):
        metadata_location = "./data/metadata.jsonl"
        if not os.path.exists(metadata_location):
            raise HTTPException(status_code=404, detail="Document not found")
        
        with open(metadata_location, "r") as f:
            for line in f:
                metadata = json.loads(line.strip())
                if metadata["file_id"] == file_id:
                    return metadata
        
        raise HTTPException(status_code=404, detail="Document not found")

    
    def extract_text(self, file_id: str):
        doc = self.get_document(file_id)
        file_path = doc["file_path"]
        
        pdf = fitz.open(file_path)
        text = ""
        
        for page in pdf:
            text += page.get_text()
            
        pdf.close()
        return text
    

doc_service = DocumentService()