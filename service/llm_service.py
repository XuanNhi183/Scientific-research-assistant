from prompt.rag_prompt import RAG_SYSTEM_PROMPT
from openai import OpenAI
import os

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_answer(self, question: str, context: str, model: str = "gpt-4.1-mini") -> str:
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

    def generate_raw(self, prompt: str, model: str = "gpt-4.1-mini") -> str:
        """Raw completion without system prompt. Used for dataset question generation."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    
    
llm_service = LLMService()