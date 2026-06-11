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
    
    def ask(self, question: str, paper_id: str | None = None, top_k: int = 5):
        query_embedding = embedding_service.embed_query(question)
        chunks = chroma_service.search(query_embedding, top_k, paper_id=paper_id)
        
        # GLOBAL CONTEXT INJECTION: Luôn luôn tiêm đoạn văn số 0 vào ngữ cảnh
        if paper_id:
            first_chunk = chroma_service.get_first_chunk(paper_id)
            if first_chunk:
                # Lọc bỏ first_chunk nếu nó đã tồn tại trong kết quả tìm kiếm
                chunks = [c for c in chunks if c.get("metadata", {}).get("chunk_index") != 0]
                # Sau đó luôn luôn nhét first_chunk lên ĐẦU danh sách
                chunks.insert(0, first_chunk)
                    
        context = self.build_context(chunks)
        answer = llm_service.generate_answer(question, context)
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            chunk_label = meta.get("chunk_index")
            if chunk_label is None:
                chunk_label = meta.get("chunk_id")
            sources.append({
                "file": meta.get("title"),
                "page": meta.get("page_start"),
                "chunk": chunk_label
            })

            print(
                f"""
    File: {meta.get('title')}
    Page: {meta.get('page_start')}
    Chunk: {chunk_label}
    Distance: {chunk.get('distance')}
    """
            )
        return {"answer": answer, "sources": sources}


rag_service = RAGService()
