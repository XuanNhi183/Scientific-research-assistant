from fastapi import FastAPI, UploadFile, HTTPException
from schemas.document import UploadResponse, DocumentContentResponse
from schemas.rag import QuestionRequest, AnswerResponse
from schemas.chunk import Chunk
from schemas.search import SearchRequest
from dotenv import load_dotenv
import uvicorn
load_dotenv()

from service.document_service import doc_service
from service.chroma_service import chroma_service
from service.rag_service import RAGService
from service.embedding_service import embedding_service
from service.chunking import get_chunks as build_chunks


app = FastAPI()

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
    rag_service = RAGService()
    return rag_service.ask(request.question, top_k=5)


@app.get("/document/{file_id}/chunks",response_model=list[Chunk])
async def get_document_chunks(file_id: str):
    return doc_service.load_chunks(file_id)


@app.get("/document/{file_id}/content", response_model=DocumentContentResponse)
async def get_document_content(file_id: str):
    content = doc_service.extract_text(file_id)
    return DocumentContentResponse(file_id=file_id, content=content)


@app.post("/search/")
async def search_documents(request: SearchRequest, top_k: int = 5):
    query_embedding = embedding_service.embed_query(request.question)
    results = chroma_service.search(query_embedding, top_k)
    return results








# http://127.0.0.1:8000/docs
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )