from service.chroma_service import chroma_service
from service.llm_service import llm_service
from service.embedding_service import embedding_service
import re

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

    def get_global_context(self, paper_id: str, query: str) -> list:
        all_chunks = chroma_service.get_all_chunks(paper_id)
        
        query_lower = query.lower()
        # 1. Keyword-based section filtering to reduce noise and tokens
        if any(kw in query_lower for kw in ["công thức", "toán học", "phương trình", "formula", "equation", "math"]):
            math_pattern = re.compile(r'\$.*?\$|\\frac|\\sum|\\int|[A-Za-z]\s*=\s*[A-Za-z0-9]', re.IGNORECASE)
            filtered = [c for c in all_chunks if math_pattern.search(c['text'])]
            if filtered:
                all_chunks = filtered
                
        elif any(kw in query_lower for kw in ["bảng", "kết quả", "hiệu suất", "table", "result", "performance"]):
            filtered = [c for c in all_chunks if any(k in c['text'].lower() for k in ["table", "result", "figure", "bảng", "hình"])]
            if filtered:
                all_chunks = filtered
                
        # 2. Token limit check to prevent surprise bills (Max ~100k tokens)
        total_tokens = sum(len(chunk['text'].split()) * 1.3 for chunk in all_chunks)
        if total_tokens > 100_000:
            print(f"[RAG] WARNING: Global context too large ({total_tokens} tokens). Falling back to priority sections.")
            # Fallback: keep first 30 chunks (usually abstract, intro) and last 10 chunks (usually conclusion)
            all_chunks = all_chunks[:30] + all_chunks[-10:]
            
        return all_chunks
    
    def ask(self, question: str, paper_id: str | None = None, top_k: int = 7):
        # 1. Classify query (Local vs Global)
        query_type = llm_service.classify_query(question)
        print(f"\n[ROUTER] Classified as: {query_type.upper()}")

        if query_type == "global" and paper_id:
            print("[RAG] Executing GLOBAL query path...")
            # Retrieve global context with smart filtering
            chunks = self.get_global_context(paper_id, question)
            context = self.build_context(chunks)
            answer = llm_service.generate_global_answer(question, context)
        else:
            print("[RAG] Executing LOCAL query path...")
            # 2. Query Translation & Optimization (Advanced RAG)
            rewrite_prompt = f"""You are a multilingual AI assistant.
Your task is to translate the user's question into a natural language English question, which will be used to search a vector database.
Keep the exact same semantic meaning and detail as the original question. Do not shorten it into keywords.
Return ONLY the English question, without quotes, explanations, or extra text.

User Question: {question}"""
            search_query = llm_service.generate_raw(rewrite_prompt)
            print(f"[RAG] Original Question: {question}")
            print(f"[RAG] Optimized English Query: {search_query}")

            # 3. Embed the English search query
            query_embedding = embedding_service.embed_query(search_query)
            chunks = chroma_service.search(query_embedding, top_k, paper_id=paper_id)
            
            # GLOBAL CONTEXT INJECTION: Luôn luôn tiêm các đoạn văn đầu bài (Chunk 0 & Chunk 1) vào ngữ cảnh
            if paper_id:
                initial_chunks = chroma_service.get_initial_chunks(paper_id, count=2)
                if initial_chunks:
                    initial_indices = {c.get("metadata", {}).get("chunk_index") for c in initial_chunks}
                    # Lọc bỏ các chunk này nếu chúng đã tồn tại trong kết quả tìm kiếm
                    chunks = [c for c in chunks if c.get("metadata", {}).get("chunk_index") not in initial_indices]
                    # Sau đó luôn luôn nhét các initial_chunks vào cuối danh sách (gần câu hỏi nhất)
                    chunks = chunks + initial_chunks
                        
            context = self.build_context(chunks)
            answer = llm_service.generate_answer(search_query, context)

            # --- HYBRID FALLBACK LOGIC ---
            if "INSUFFICIENT_INFORMATION" in answer and paper_id:
                print("\n[FALLBACK] Local search failed (Insufficient Info). Triggering GLOBAL FALLBACK...")
                chunks = self.get_global_context(paper_id, question)
                context = self.build_context(chunks)
                answer = llm_service.generate_global_answer(question, context)
                query_type = "global_fallback"
            else:
                # Dịch câu trả lời của local sang tiếng Việt bằng gpt-4o-mini mà không chỉnh sửa nội dung
                print("\n[RAG] Translating local output to Vietnamese with GPT-4o-mini...")
                answer = llm_service.translate_answer(question, answer)
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
            if query_type == "local":
                print(
                    f"""
    File: {meta.get('title')}
    Page: {meta.get('page_start')}
    Chunk: {chunk_label}
    Distance: {chunk.get('distance', 0.0)}
    """
                )
            
        return {"answer": answer, "sources": sources, "query_type": query_type}


rag_service = RAGService()
