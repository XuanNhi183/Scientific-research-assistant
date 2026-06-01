from openai import OpenAI
import os
from app.schemas.chunk import Chunk

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_query(self, query: str, 
                   model: str ="text-embedding-3-small") -> list[float]: #1536 float
        response = self.client.embeddings.create(
            input=query,
            model=model
        )
        return response.data[0].embedding
    
    def embed_chunks(self, chunks: list[Chunk], model: str = "text-embedding-3-small") -> list[Chunk]:

        texts = [chunk.text for chunk in chunks]

        response = self.client.embeddings.create(
            model=model,
            input=texts
        )

        for chunk, embedding_data in zip(chunks, response.data):
            chunk.embedding = embedding_data.embedding

        return chunks
        
    
embedding_service = EmbeddingService()