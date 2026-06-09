import json
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


from prompt.single_hop_prompt import SINGLE_HOP_PROMPT
from prompt.multi_hop_prompt import MULTI_HOP_PROMPT
from prompt.question_validator_prompt import QUESTION_VALIDATOR_PROMPT
from prompt.answer_validator_prompt import ANSWER_VALIDATOR_PROMPT
from prompt.unanswerable_prompt import UNANSWERABLE_PROMPT

class QAGenerator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.6) -> str:
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def validate_question(self, question: str, context: str) -> dict:
        """
        Validates a generated question using the two-stage rubric (evidence extraction + quality scoring).
        Returns a dict with keys: valid, reason, difficulty, and per-criterion scores.
        """
        user_msg = f"[Context]\n{context}\n\n[Question]\n{question}"
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": QUESTION_VALIDATOR_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
            # Enforce: any single criterion FAIL → override valid to False
            fail_criteria = [
                result.get("q1_extractiveness") == "FAIL",
                result.get("q2_specificity") == "FAIL",
                result.get("q3_groundedness") == "FAIL",
                result.get("q4_answerability") == "FAIL",
                result.get("q5_dependency") == "FAIL",
            ]
            if any(fail_criteria):
                result["valid"] = False
            return result
        except json.JSONDecodeError:
            return {"valid": False, "reason": "JSON decode error", "difficulty": "HARD"}


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def validate_answer(self, question: str, answer: str, context: str) -> dict:
        """
        Validates a generated answer using the answer quality rubric.
        Returns a dict with keys: valid, reason, and per-criterion scores.
        """
        user_msg = (
            f"[Context]\n{context}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Answer]\n{answer}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANSWER_VALIDATOR_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
            # Enforce: any single criterion FAIL → override valid to False
            fail_criteria = [
                result.get("a1_faithfulness") == "FAIL",
                result.get("a2_completeness") == "FAIL",
                result.get("a3_non_extractiveness") == "FAIL",
                result.get("a4_coherence") == "FAIL",
                result.get("a5_language") == "FAIL",
            ]
            if any(fail_criteria):
                result["valid"] = False
            return result
        except json.JSONDecodeError:
            return {"valid": False, "reason": "JSON decode error"}


    def generate_single_hop(self,chunk: str,title: str,
                            section: str = "Unknown",n: int = 2,) -> list[str]:
        prompt = SINGLE_HOP_PROMPT.format(n=n, title=title, section=section, chunk=chunk)
        raw = self._call_llm(prompt)
        questions = self._parse_json_list(raw)
 
        valid_qs = []
        for q in questions:
            val = self.validate_question(q, chunk)
            if val.get("valid", False):
                valid_qs.append(q)
            else:
                print(f"  [Q-Validator] Rejected single-hop: {q!r}")
                print(f"    Reason: {val.get('reason')} | "
                      f"Extractive={val.get('q1_extractiveness')} | "
                      f"Specific={val.get('q2_specificity')} | "
                      f"Grounded={val.get('q3_groundedness')} | "
                      f"Answerable={val.get('q4_answerability')}")
        return valid_qs

    def generate_multi_hop(self,chunk_a: str,chunk_b: str,section_a: str,
                            section_b: str,title: str,n: int = 1,) -> list[str]:
        prompt = MULTI_HOP_PROMPT.format(
            n=n,
            title=title,
            chunk_a=chunk_a,
            chunk_b=chunk_b,
            section_a=section_a,
            section_b=section_b,
        )
        raw = self._call_llm(prompt)
        questions = self._parse_json_list(raw)
 
        # Generator returning [] is valid — means no genuine multi-hop bridge exists
        if not questions:
            print("  [Multi-hop] Generator returned empty array — no valid bridge found.")
            return []
 
        context = f"[Chunk A]\n{chunk_a}\n\n[Chunk B]\n{chunk_b}"
        valid_qs = []
        for q in questions:
            val = self.validate_question(q, context)
            if val.get("valid", False):
                valid_qs.append(q)
            else:
                print(f"  [Q-Validator] Rejected multi-hop: {q!r}")
                print(f"    Reason: {val.get('reason')} | "
                      f"Dependency={val.get('q5_dependency')} | "
                      f"Specific={val.get('q2_specificity')} | "
                      f"Extractive={val.get('q1_extractiveness')}")
        return valid_qs

    def generate_unanswerable_question(self, chunk: str, title: str, section: str = "Unknown") -> list[str]:
        prompt = UNANSWERABLE_PROMPT.format(n=1, title=title, section=section, chunk=chunk)
        raw = self._call_llm(prompt)
        questions = self._parse_json_list(raw)
        return questions

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_answer(
        self,
        question: str,
        formatted_context: str,
        system_prompt: str,
        validate: bool = True,
        question_type: str = "positive",
    ) -> str | None:
        """
        Generates an answer and optionally validates it with ANSWER_VALIDATOR_PROMPT.
        Returns the answer string if valid, or None if validation fails.
        Set validate=False to skip answer validation (e.g. for INSUFFICIENT_INFORMATION cases).
        """
        # Dynamic length and synthesis rules
        behavior_rules = "\n\nCRITICAL RULE: Your answer MUST be entirely in English, regardless of the context language.\nNEVER reference the chunks directly (e.g. do not say 'Chunk 1 states'). Synthesize seamlessly."
        
        if question_type in ["positive", "noisy_positive"]:
            behavior_rules += "\nRULE: Answer concisely and directly without fluff."
        elif question_type == "multi_chunk":
            behavior_rules += "\nRULE: Provide a comprehensive synthesis. Explain the logical relationship bridging the concepts from the context."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt + behavior_rules},
                {"role": "user", "content": f"[Context]\n{formatted_context}\n\n[Question]\n{question}"}
            ],
            max_tokens=512,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
 
        if not validate:
            return answer
            
        MIN_WORDS = 12
        if answer != "INSUFFICIENT_INFORMATION" and len(answer.split()) < MIN_WORDS:
            print(f"  [A-Validator] Rejected short answer ({len(answer.split())} words) for: {question!r}")
            return None
 
        val = self.validate_answer(question, answer, formatted_context)
        if val.get("valid", False):
            return answer
        else:
            print(f"  [A-Validator] Rejected answer for: {question!r}")
            print(f"    Reason: {val.get('reason')} | "
                  f"Faithful={val.get('a1_faithfulness')} | "
                  f"Complete={val.get('a2_completeness')} | "
                  f"NonExtract={val.get('a3_non_extractiveness')} | "
                  f"Coherent={val.get('a4_coherence')} | "
                  f"English={val.get('a5_language')}")
            return None