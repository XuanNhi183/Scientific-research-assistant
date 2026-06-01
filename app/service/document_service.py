from fastapi import UploadFile, HTTPException
import os
from uuid import uuid4
import json
import fitz

from app.schemas.chunk import Chunk, ChunkMetadata
 

class DocumentService:
    async def upload_file(self, file: UploadFile):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are allowed."
                )
            
        upload_location = "./data/uploads/"
        chroma_location = "./data/chroma"
        os.makedirs(upload_location, exist_ok=True)
        os.makedirs(chroma_location, exist_ok=True)

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

    def extract_pages(self, file_id: str):
        doc = self.get_document(file_id)
        file_path = doc["file_path"]

        pdf = fitz.open(file_path)
        pages = []
        for page_index, page in enumerate(pdf):
            page_text = page.get_text()
            if page_text.strip():
                pages.append({
                    "page_number": page_index + 1,
                    "text": page_text,
                })

        pdf.close()
        return pages
    
    
    def chunking(self, text: str, chunk_size:int, overlap:int):
        if overlap >= chunk_size:
                raise ValueError("Overlap must be smaller than chunk size.")
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = (" ".join(words[start:end]))
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks

    def save_chunks(self, file_id: str, chunks: list[Chunk]):
        chunks_location = f"./data/chunks/{file_id}.jsonl"
        os.makedirs(os.path.dirname(chunks_location), exist_ok=True)

        with open(chunks_location, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump()) + "\n")

        return {
            "file_id": file_id,
            "chunk_count": len(chunks),
            "chunks_location": chunks_location
        }
    
    
    def get_chunks(self, file_id: str, chunk_size: int, overlap: int):
        doc = self.get_document(file_id)
        pages = self.extract_pages(file_id)
        results = []
        chunk_index = 0

        for page in pages:
            page_chunks = self.chunking(page["text"], chunk_size, overlap)
            for chunk_text in page_chunks:
                metadata = ChunkMetadata(
                    paper_id=file_id,
                    title=doc["filename"],
                    page_start=page["page_number"],
                    page_end=page["page_number"],
                    source_pdf=doc["file_path"],
                )
                results.append(
                    Chunk(
                        chunk_id=str(uuid4()),
                        chunk_index=chunk_index,
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                        metadata=metadata,
                    )
                )
                chunk_index += 1
        return results

    
    def process_document(self, file_id: str, chunk_size: int, overlap: int):
        chunks = self.get_chunks(file_id, chunk_size, overlap)
        self.save_chunks(file_id, chunks)
        return chunks
    
    
    def load_chunks(self, file_id: str):
        chunk_file = f"./data/chunks/{file_id}.jsonl"

        if not os.path.exists(chunk_file):
            raise HTTPException(
                status_code=404,
                detail="Chunks not found"
            )

        chunks = []

        with open(chunk_file, "r") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))

        return chunks

    
    
    
doc_service = DocumentService()