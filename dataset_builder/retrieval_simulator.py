import re
import random
from typing import Optional

from dataset_builder.qa_generator import QAGenerator
from schemas.chunk import Chunk


BAD_PATTERNS = [
    "not specified", "not mentioned", "does not provide",
    "cannot be determined", "cannot be concluded", "not enough information",
    "insufficient information", "not available in the context",
    "does not explicitly state", "context does not mention",
    "is not explicitly stated", "the paper does not", "i could not find",
]

SPECULATIVE_TERMS = [
    "it is likely that", "it is probably", "we can assume",
    "one might speculate", "it seems reasonable to conclude",
    "outside the scope of the provided context",
    "not explicitly covered", "beyond what is stated",
]


def is_bad_answer(answer: str) -> bool:
    a = answer.lower().strip()
    return any(p in a for p in BAD_PATTERNS)


def is_speculative(answer: str) -> bool:
    a = answer.lower().strip()
    return any(t in a for t in SPECULATIVE_TERMS)


def format_context(chunks: list[str]) -> str:
    return "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks))


def _is_citation_chunk(text: str) -> bool:
    """
    Detect chunks that are primarily citation/bibliography content with no
    technical substance. Multi-hop questions over these always return
    INSUFFICIENT_INFORMATION, corrupting the multi_chunk label.

    Two independent detectors — either one firing is sufficient.

    Detector A: >25% of lines start with [N] bracket references.
    Detector B: >70% of lines are short (<80 chars) AND >10% of lines
                contain venue/year keywords (Proceedings, arXiv, ICML, etc.)
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    n = len(lines)

    # Detector A: numbered bracket reference list
    bracket_refs = sum(1 for l in lines if re.match(r'^\s*\[\d+\]', l))
    if bracket_refs / n > 0.25:
        return True

    # Detector B: un-bracketed bibliography (author/title/venue lines)
    short_lines = sum(1 for l in lines if len(l.strip()) < 80)
    venue_lines = sum(1 for l in lines if re.search(
        r'\b(Proceedings|arXiv|Journal|Conference|Workshop|Transactions|'
        r'CVPR|ICLR|NeurIPS|ICML|ACL|EMNLP|NAACL|AAAI|IJCAI|ECCV|ICCV)\b',
        l, re.IGNORECASE
    ))
    if short_lines / n > 0.70 and venue_lines / n > 0.10:
        return True

    return False


# ── RetrievalSimulator ─────────────────────────────────────────────────────────

class RetrievalSimulator:
    def __init__(self, qa_generator: QAGenerator, system_prompt: str):
        self.qa = qa_generator
        self.system_prompt = system_prompt

    def _get_hierarchical_distractors(self, target: Chunk, paper_chunks: list[Chunk], distractor_pool: list[Chunk], k: int) -> list[str]:
        """
        Selects k distractors hierarchically:
        1. Same paper, different section
        2. Same paper, same section, different chunk
        3. Similar paper (same category)
        4. Random paper
        """
        distractors = []

        # Level 1: same paper, different section
        pool_l1 = [c for c in paper_chunks if c.metadata.section != target.metadata.section and c.chunk_id != target.chunk_id]
        if pool_l1:
            sampled = random.sample(pool_l1, min(k, len(pool_l1)))
            distractors.extend([c.text for c in sampled])

        # Level 2: same paper, same section, different chunk
        if len(distractors) < k:
            pool_l2 = [c for c in paper_chunks if c.metadata.section == target.metadata.section and c.chunk_id != target.chunk_id]
            if pool_l2:
                sampled = random.sample(pool_l2, min(k - len(distractors), len(pool_l2)))
                distractors.extend([c.text for c in sampled])

        # Level 3: similar paper (same category)
        if len(distractors) < k and distractor_pool:
            target_cat = target.metadata.category
            pool_l3 = [c for c in distractor_pool if c.metadata.category == target_cat]
            if pool_l3:
                sampled = random.sample(pool_l3, min(k - len(distractors), len(pool_l3)))
                distractors.extend([c.text for c in sampled])

        # Level 4: any remaining paper
        if len(distractors) < k and distractor_pool:
            pool_l4 = [c for c in distractor_pool if c.metadata.category != target.metadata.category]
            if pool_l4:
                sampled = random.sample(pool_l4, min(k - len(distractors), len(pool_l4)))
                distractors.extend([c.text for c in sampled])

        return distractors

    def build_sample(
        self,
        paper_id: str,
        title: str,
        paper_chunks: list[Chunk],
        distractor_pool: list[Chunk],
    ) -> Optional[dict]:
        if len(paper_chunks) < 2:
            return None

        roll = random.random()
        target = random.choice(paper_chunks)
        distractor_texts = self._get_hierarchical_distractors(target, paper_chunks, distractor_pool, 3)

        if roll >= 0.60 and len(distractor_texts) == 0:
            return None

        if roll < 0.35:
            return self._easy(paper_id, title, target)
        elif roll < 0.60:
            return self._medium_multihop(paper_id, title, paper_chunks)
        elif roll < 0.75:
            return self._medium_noisy(paper_id, title, target, distractor_texts[:1])
        elif roll < 0.90:
            return self._hard_noisy(paper_id, title, target, distractor_texts[:2])
        else:
            if len(distractor_texts) < 3:
                return self._hard_noisy(paper_id, title, target, distractor_texts)
            return self._hard_insufficient(paper_id, title, distractor_texts[:3])

    # ── private builders ──────────────────────────────────────────────────────

    def _easy(self, paper_id, title, target) -> Optional[dict]:
        questions = self.qa.generate_single_hop(
            target.text, title, section=target.metadata.section or "Unknown", n=1
        )
        if not questions:
            return None

        context = format_context([target.text])
        answer, difficulty = self._get_answer(questions[0], context, fallback_difficulty="EASY", question_type="positive")
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

        # Filter out citation/bibliography chunks before picking — they have no
        # technical content and cause INSUFFICIENT_INFORMATION answers with multi_chunk label.
        content_chunks = [c for c in paper_chunks if not _is_citation_chunk(c.text)]
        if len(content_chunks) < 2:
            print(f"  [simulator] multi-hop skipped: <2 non-citation chunks available")
            return None

        # Prefer chunks from DIFFERENT sections for genuine cross-section synthesis
        sections = {}
        for c in content_chunks:
            sec = c.metadata.section or "Unknown"
            sections.setdefault(sec, []).append(c)

        if len(sections) >= 2:
            sec_a, sec_b = random.sample(list(sections.keys()), 2)
            chunk_a = random.choice(sections[sec_a])
            chunk_b = random.choice(sections[sec_b])
        else:
            chunk_a, chunk_b = random.sample(content_chunks, 2)

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
        answer, difficulty = self._get_answer(
            questions[0], context, fallback_difficulty="MEDIUM", question_type="multi_chunk"
        )

        # Discard instead of saving with wrong label — INSUF here means the question
        # wasn't genuinely multi-hop, saving it corrupts the multi_chunk signal.
        if answer == "INSUFFICIENT_INFORMATION":
            print(f"  [simulator] multi-hop discarded: answer was INSUFFICIENT_INFORMATION")
            return None

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
        answer, _ = self._get_answer(questions[0], context, fallback_difficulty="MEDIUM", question_type="noisy_positive")

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
        answer, _ = self._get_answer(questions[0], context, fallback_difficulty="HARD", question_type="noisy_positive")

        return self._make_sample(
            paper_id, "HARD", "noisy_positive",
            questions[0], context, answer,
            section=target.metadata.section,
        )

    def _hard_insufficient(self, paper_id, title, distractors) -> Optional[dict]:
        context = format_context(distractors)

        base_distractor = distractors[0]
        questions = self.qa.generate_unanswerable_question(
            chunk=base_distractor,
            title=title,
            section="Unknown",
        )
        if not questions:
            return None

        question = questions[0]
        return self._make_sample(
            paper_id, "HARD", "insufficient",
            question, context,
            "INSUFFICIENT_INFORMATION",
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_answer(self, question: str, context: str, fallback_difficulty: str, question_type: str) -> tuple[str, str]:
        answer = self.qa.generate_answer(question, context, self.system_prompt, question_type=question_type)
        if answer is None:
            return "INSUFFICIENT_INFORMATION", "HARD"
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