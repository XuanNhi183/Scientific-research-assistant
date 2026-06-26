from prompt.rag_prompt import RAG_SYSTEM_PROMPT
from prompt.router_prompt import ROUTER_PROMPT
from openai import OpenAI
import os

class LLMService:
    def __init__(self):
        # Client gốc dùng cho các tác vụ phức tạp (như tóm tắt JSON)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Client trỏ về Ollama chạy ở máy Windows hoặc Colab
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.local_client = OpenAI(
            base_url=ollama_url,
            api_key="ollama",
            timeout=600.0,
            default_headers={"ngrok-skip-browser-warning": "69420"}
        )
        
    def generate_answer(self, question: str, context: str, model: str = "qwen2.5-rag") -> str:
        user_prompt = f"""
        [Context]
        {context}
        
        [Question]
        {question}
        """
        response = self.local_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", 
                 "content": RAG_SYSTEM_PROMPT},
                {"role": "user", 
                 "content": user_prompt}
            ],
            temperature=0.1, # Đặt nhiệt độ thấp để tránh bịa chuyện
        )
        return response.choices[0].message.content.strip()

    def classify_query(self, query: str, model: str = "qwen2.5-rag") -> str:
        """Classifies a query as either LOCAL or GLOBAL."""
        response = self.local_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.0, # Deterministic classification
        )
        result = response.choices[0].message.content.strip().upper()
        
        if "GLOBAL" in result:
            return "global"
        elif "LOCAL" in result:
            return "local"
        else:
            return "local" # fallback

    def generate_global_answer(self, question: str, context: str, model: str = "gpt-4o-mini") -> str:
        """Generates answer using GPT-4o-mini to handle large global contexts."""
        user_prompt = f"""
        [Context]
        {context}
        
        [Question]
        {question}
        """
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", 
                 "content": RAG_SYSTEM_PROMPT},
                {"role": "user", 
                 "content": user_prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def generate_raw(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Raw completion without system prompt. Used for dataset question generation."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    
    def generate_paper_analysis(self, raw_text: str, title: str, model: str = "gpt-4o-mini") -> str:
        system_prompt = """Bạn là một chuyên gia AI phân tích cấu trúc bài báo khoa học.
Nhiệm vụ của bạn là bóc tách cấu trúc văn bản thô thành một JSON hoàn hảo.
JSON BẮT BUỘC phải có đúng các key sau:
- "title": (string) Tiêu đề bài báo
- "authors": (string) Danh sách tác giả
- "year": (string) Năm xuất bản
- "abstract": (string) Tóm tắt (150-200 từ bằng Tiếng Việt súc tích)
- "sections": (array of objects) Mỗi object có "title" (tên phần) và "content" (nội dung chi tiết tiếng Việt)
- "metrics": (object) Chứa "novelty" (int 0-100), "complexity" (string: 'Cơ bản', 'Trung bình', 'Chuyên sâu'), "readingTime" (int phút), "citations" (int)
- "keyFindings": (array of strings) 3-5 phát hiện chính
- "glossary": (array of objects) Mỗi object có "term" (thuật ngữ) và "definition" (giải nghĩa tiếng Việt)
"""
        user_prompt = f"Tiêu đề dự kiến: {title}\n\nVăn bản bài báo:\n{raw_text[:25000]}" # Limit text to avoid context overload
        
        response = self.openai_client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def translate_answer(self, question: str, local_answer: str) -> str:
        """Translates the local model's answer into the user's target language (Vietnamese) without modifying its content."""
        system_prompt = """You are a professional translator.
Your sole task is to translate the provided Answer into Vietnamese, matching the language of the User's Question.

Guidelines:
1. Translate the Answer exactly as it is. Do not add, remove, or modify any facts, numbers, metrics, or meanings.
2. Keep scientific terms (e.g. BERT, GCN, IGLU, MRR, P@1) in English.
3. If the Answer is already in Vietnamese, return it unchanged.
4. Output ONLY the translated Vietnamese text. Do not add any introduction, explanation, or notes.
"""
        user_prompt = f"""
[User Question]
{question}

[Answer to Translate]
{local_answer}
"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Deterministic translation
        )
        return response.choices[0].message.content.strip()

llm_service = LLMService()
