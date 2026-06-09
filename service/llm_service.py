from prompt.rag_prompt import RAG_SYSTEM_PROMPT
from openai import OpenAI
import os

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_answer(self, question: str, context: str, model: str = "gpt-4o-mini") -> str:
        user_prompt = f"""
        [Question]
        {question}
        
        [Context]
        {context}
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", 
                 "content": RAG_SYSTEM_PROMPT},
                {"role": "user", 
                 "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def generate_raw(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Raw completion without system prompt. Used for dataset question generation."""
        response = self.client.chat.completions.create(
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
        
        response = self.client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

llm_service = LLMService()
