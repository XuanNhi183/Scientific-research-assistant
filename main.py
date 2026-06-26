from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from schemas.document import UploadResponse, DocumentContentResponse
from schemas.rag import QuestionRequest, AnswerResponse, AnalyzeRequest
from schemas.chunk import Chunk
from dotenv import load_dotenv
import uvicorn
load_dotenv()

from service.document_service import doc_service
from service.chroma_service import chroma_service
from service.rag_service import rag_service
from service.embedding_service import embedding_service
from service.chunking import get_chunks as build_chunks
from service.llm_service import llm_service
import json


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn frontend kết nối
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_processed_File/", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )
    try:
        result = await doc_service.upload_file(file)
        chunks = build_chunks(
            result["file_id"],
            chunk_size=700,
            overlap=150
        )
        for chunk in chunks[:10]:
            print(
                chunk.metadata.page_start,
                chunk.metadata.page_end,
                chunk.metadata.char_start,
                chunk.metadata.char_end,
            )
        doc_service.save_chunks(result["file_id"], chunks)
        chunks = embedding_service.embed_chunks(chunks)
        chroma_service.add_chunks(chunks)
        print(
            f"Total chunks in ChromaDB: {chroma_service.collection.count()}"
        )
        return UploadResponse(**result)
   
    except HTTPException:
        raise 
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question/", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    return rag_service.ask(request.question, request.paper_id, top_k=7)


@app.get("/document/{file_id}/chunks",response_model=list[Chunk])
async def get_document_chunks(file_id: str):
    return doc_service.load_chunks(file_id)


@app.get("/document/{file_id}/pdf")
async def get_document_pdf(file_id: str):
    file_path = f"./data/uploads/{file_id}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/document/{file_id}/content", response_model=DocumentContentResponse)
async def get_document_content(file_id: str):
    content = doc_service.extract_text(file_id)
    return DocumentContentResponse(file_id=file_id, content=content)


@app.post("/search/")
async def search_documents(request: QuestionRequest, top_k: int = 5):
    query_embedding = embedding_service.embed_query(request.question)
    results = chroma_service.search(query_embedding, top_k)
    return results

@app.post("/analyze_paper/")
async def analyze_paper(request: AnalyzeRequest):
    try:
        json_str = llm_service.generate_paper_analysis(request.rawText, request.title)
        # Parse into dict to return as JSON response
        return json.loads(json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








# http://127.0.0.1:8000/docs
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )