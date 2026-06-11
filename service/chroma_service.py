import chromadb
from schemas.chunk import Chunk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


class ChromaService:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(name=collection_name)


    def add_chunks(self, chunks: list[Chunk]):
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas=[
            {
                "paper_id": chunk.metadata.paper_id,
                "title": chunk.metadata.title,
                "section": chunk.metadata.section,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.metadata.page_start,
                "page_end": chunk.metadata.page_end,
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    
    def search(self, query_embedding, top_k=5, paper_id: str | None = None):
        where_clause = None
        if paper_id:
            where_clause = {"paper_id": paper_id}
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        chunks = []

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": distance
            })

        return chunks
    
    def get_first_chunk(self, paper_id: str):
        # Trích xuất đoạn văn đầu tiên (thường là Abstract/Giới thiệu/Tác giả)
        # Sửa lỗi ChromaDB: Phải dùng $and khi có nhiều hơn 1 điều kiện
        results = self.collection.get(
            where={"$and": [{"paper_id": paper_id}, {"chunk_index": 0}]}
        )
        
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return None
            
        return {
            "text": results["documents"][0],
            "metadata": results["metadatas"][0],
            "distance": 0.0 # Bỏ qua distance vì đây là trích xuất trực tiếp
        }

chroma_service = ChromaService(collection_name="document_chunks")