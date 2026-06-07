"""
RetrievalSimulator: Simulates RAG retrieval scenarios to generate diverse SFT samples.

Distribution (matches target 40% EASY / 35% MEDIUM / 25% HARD):

  roll < 0.40  → EASY:   1 target chunk            (Positive QA)
  roll < 0.60  → MEDIUM: 2 chunks, multi-hop Q     (Multi-Chunk QA)
  roll < 0.75  → MEDIUM: target + 1 distractor     (Noisy Positive QA)
  roll < 0.90  → HARD:   target + 2 distractors    (Noisy Positive QA)
  roll < 1.00  → HARD:   3 distractors only        (Insufficient Info QA)
"""

import random
from typing import Optional

from dataset.qa_generator import QAGenerator


# ── Answer quality guards ──────────────────────────────────────────────────────

BAD_PATTERNS = [
    "not specified", "not mentioned", "does not provide",
    "cannot be determined", "cannot be concluded", "not enough information",
    "insufficient information", "not available in the context",
    "does not explicitly state", "context does not mention",
    "is not explicitly stated", "the paper does not", "i could not find",
]

SPECULATIVE_TERMS = [
    "likely", "probably", "suggests", "implies", "may indicate",
    "can be inferred", "implicitly", "appears to", "is suggested",
]


def is_bad_answer(answer: str) -> bool:
    a = answer.lower().strip()
    return any(p in a for p in BAD_PATTERNS)


def is_speculative(answer: str) -> bool:
    a = answer.lower().strip()
    return any(t in a for t in SPECULATIVE_TERMS)


def format_context(chunks: list[str]) -> str:
    return "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))


# ── RetrievalSimulator ─────────────────────────────────────────────────────────

class RetrievalSimulator:
    def __init__(self, qa_generator: QAGenerator, system_prompt: str):
        self.qa = qa_generator
        self.system_prompt = system_prompt

    def build_sample(
        self,
        paper_id: str,
        title: str,
        paper_chunks: list,       # list[Chunk] from THIS paper
        distractor_pool: list[str],  # chunk texts from OTHER papers
    ) -> Optional[dict]:
        """
        Roll a random number to decide the sample type and build it.
        Returns a sample dict or None if generation fails.
        """
        if len(paper_chunks) < 2 or len(distractor_pool) < 3:
            return None

        roll = random.random()
        target = random.choice(paper_chunks)
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))

        if roll < 0.40:
            return self._easy(paper_id, title, target)
        elif roll < 0.60:
            return self._medium_multihop(paper_id, title, paper_chunks)
        elif roll < 0.75:
            return self._medium_noisy(paper_id, title, target, distractors[:1])
        elif roll < 0.90:
            return self._hard_noisy(paper_id, title, target, distractors[:2])
        else:
            return self._hard_insufficient(paper_id, distractors[:3])

    # ── private builders ──────────────────────────────────────────────────────

    def _easy(self, paper_id, title, target) -> Optional[dict]:
        questions = self.qa.generate_single_hop(
            target.text, title, section=target.metadata.section or "Unknown", n=1
        )
        if not questions:
            return None

        context = format_context([target.text])
        answer, difficulty = self._get_answer(questions[0], context, fallback_difficulty="EASY")
        if answer != "INSUFFICIENT_INFORMATION" and len(answer.split()) < 10:
            return None

        return self._make_sample(
            paper_id, difficulty, "positive",
            questions[0], context, answer,
            section=target.metadata.section,
        )

    def _medium_multihop(self, paper_id, title, paper_chunks) -> Optional[dict]:
        if len(paper_chunks) < 2:
            return None

        # Prefer chunks from DIFFERENT sections for richer multi-hop
        sections = {}
        for c in paper_chunks:
            sec = c.metadata.section or "Unknown"
            sections.setdefault(sec, []).append(c)

        if len(sections) >= 2:
            sec_a, sec_b = random.sample(list(sections.keys()), 2)
            chunk_a = random.choice(sections[sec_a])
            chunk_b = random.choice(sections[sec_b])
        else:
            chunk_a, chunk_b = random.sample(paper_chunks, 2)

        questions = self.qa.generate_multi_hop(
            chunk_a.text, chunk_b.text,
            chunk_a.metadata.section or "Section A",
            chunk_b.metadata.section or "Section B",
            title, n=1,
        )
        if not questions:
            return None

        chunks = [chunk_a.text, chunk_b.text]
        random.shuffle(chunks)
        context = format_context(chunks)
        answer, difficulty = self._get_answer(questions[0], context, fallback_difficulty="MEDIUM")

        return self._make_sample(paper_id, difficulty, "multi_chunk", questions[0], context, answer)

    def _medium_noisy(self, paper_id, title, target, distractors) -> Optional[dict]:
        questions = self.qa.generate_single_hop(
            target.text, title, section=target.metadata.section or "Unknown", n=1
        )
        if not questions:
            return None

        chunks = [target.text] + [d for d in distractors]
        random.shuffle(chunks)
        context = format_context(chunks)
        answer, _ = self._get_answer(questions[0], context, fallback_difficulty="MEDIUM")

        return self._make_sample(
            paper_id, "MEDIUM", "noisy_positive",
            questions[0], context, answer,
            section=target.metadata.section,
        )

    def _hard_noisy(self, paper_id, title, target, distractors) -> Optional[dict]:
        questions = self.qa.generate_single_hop(
            target.text, title, section=target.metadata.section or "Unknown", n=1
        )
        if not questions:
            return None

        chunks = [target.text] + distractors
        random.shuffle(chunks)
        context = format_context(chunks)
        answer, _ = self._get_answer(questions[0], context, fallback_difficulty="HARD")

        return self._make_sample(
            paper_id, "HARD", "noisy_positive",
            questions[0], context, answer,
            section=target.metadata.section,
        )

    def _hard_insufficient(self, paper_id, distractors) -> dict:
        context = format_context(distractors)
        return self._make_sample(
            paper_id, "HARD", "insufficient",
            "What is the main contribution of this paper?",
            context,
            "INSUFFICIENT_INFORMATION",
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_answer(self, question: str, context: str, fallback_difficulty: str) -> tuple[str, str]:
        """Generate answer and return (answer, difficulty)."""
        answer = self.qa.generate_answer(question, context, self.system_prompt)
        if is_bad_answer(answer) or is_speculative(answer):
            return "INSUFFICIENT_INFORMATION", "HARD"
        return answer, fallback_difficulty

    def _make_sample(
        self,
        paper_id: str,
        difficulty: str,
        question_type: str,
        question: str,
        context: str,
        answer: str,
        section: str | None = None,
    ) -> dict:
        return {
            "paper_id": paper_id,
            "question_type": question_type,
            "difficulty": difficulty,
            "section": section,
            "messages": [
                {"role": "system", "content": self.system_prompt.strip()},
                {"role": "user", "content": f"[Context]\n{context}\n\n[Question]\n{question}"},
                {"role": "assistant", "content": answer},
            ],
        }
