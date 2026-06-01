from service.chroma_service import chroma_service
from service.llm_service import llm_service
from service.embedding_service import embedding_service

class RAGService:
    def build_context(self, chunks):

        contexts = []

        for chunk in chunks:
            meta = chunk["metadata"]
            contexts.append(
                f"""
                [Source]
                [Page {meta['page_start']}]
                [File: {meta['title']}]
                {chunk['text']}
                """
            )

        return "\n\n".join(contexts)
    
    def ask(self, question: str, top_k: int = 5):
        query_embedding = embedding_service.embed_query(question)
        chunks = chroma_service.search(query_embedding, top_k)
        context = self.build_context(chunks)
        answer = llm_service.generate_answer(question, context)
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "file": meta.get("title"),
                "page": meta.get("page_start")
            })

            print(
                f"""
    File: {meta.get('title')}
    Page: {meta.get('page_start')}
    Chunk: {meta.get('chunk_index')}
    Distance: {chunk.get('score')}
    """
            )
        return {"answer": answer, "sources": sources}

rag_service = RAGService()