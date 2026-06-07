"""
QAGenerator: Generates questions and answers using the OpenAI API.

Two question modes:
- Single-hop: question answerable from one chunk alone.
- Multi-hop:  question requiring synthesis across two chunks from different sections.

Answer generation reuses RAG_SYSTEM_PROMPT from prompt/rag_prompt.py.
"""

import json
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


SINGLE_HOP_PROMPT = """\
You are a scientific question generator.

Generate {n} diverse questions that can be fully answered from the following text chunk.

Paper Title: {title}

Chunk (Section: {section}):
{chunk}

Requirements:
- Each question must be answerable from this chunk alone — no external knowledge.
- Vary the question type. Choose from:
  Definition, Methodology, Motivation, Assumption, Limitation,
  Advantage, Disadvantage, Contribution, Experimental Result, Future Work.
- Be specific. Avoid generic templates like "What is X?" or "How does X improve Y?".
- Write in English.

Return ONLY a JSON array of question strings. No explanation, no markdown.
Example: ["question1", "question2"]"""


MULTI_HOP_PROMPT = """\
You are a scientific question generator specializing in multi-hop reasoning.

Generate {n} questions that require combining information from BOTH chunks below to answer fully.

Paper Title: {title}

[Chunk A — Section: {section_a}]
{chunk_a}

[Chunk B — Section: {section_b}]
{chunk_b}

Requirements:
- The answer CANNOT come from a single chunk alone.
- Focus on cross-chunk relationships such as:
  - How experimental results validate the proposed method.
  - What limitations are revealed by the experimental findings.
  - Comparison between approaches described in different sections.
  - How assumptions in one section affect conclusions in another.
- No speculation. Questions must be fully answerable from both chunks combined.
- Write in English.

Return ONLY a JSON array of question strings. No explanation, no markdown.
Example: ["question1"]"""


class QAGenerator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.9) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_json_list(self, raw: str) -> list[str]:
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [q for q in result if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            pass
        return []

    def generate_single_hop(
        self,
        chunk: str,
        title: str,
        section: str = "Unknown",
        n: int = 2,
    ) -> list[str]:
        prompt = SINGLE_HOP_PROMPT.format(n=n, title=title, section=section, chunk=chunk)
        raw = self._call_llm(prompt)
        return self._parse_json_list(raw)

    def generate_multi_hop(
        self,
        chunk_a: str,
        chunk_b: str,
        section_a: str,
        section_b: str,
        title: str,
        n: int = 1,
    ) -> list[str]:
        prompt = MULTI_HOP_PROMPT.format(
            n=n,
            title=title,
            chunk_a=chunk_a,
            chunk_b=chunk_b,
            section_a=section_a,
            section_b=section_b,
        )
        raw = self._call_llm(prompt)
        return self._parse_json_list(raw)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_answer(self, question: str, formatted_context: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[Context]\n{formatted_context}\n\n[Question]\n{question}"},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
