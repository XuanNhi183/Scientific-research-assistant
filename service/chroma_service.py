import chromadb
from schemas.chunk import Chunk

class ChromaService:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path="./data/chroma")
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
        
    
    def search(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
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
                "score": distance
            })

        return chunks
    
chroma_service = ChromaService(collection_name="document_chunks")