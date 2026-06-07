"""
DatasetBuilder: Orchestrates the full dataset generation pipeline.

Flow:
  Kaggle arxiv metadata (paper_id, title)
    → ArxivDownloader (download PDF)
    → extract_sections()       ← reuses service/chunking.py
    → chunk_sections()         ← reuses service/chunking.py
    → RetrievalSimulator       ← builds EASY/MEDIUM/HARD samples
    → write JSONL

Chunk params match production: chunk_size=700, overlap=150.
"""

import os
import json
import time
import random
from collections import Counter

import pandas as pd

from dataset_builder.arxiv_downloader import ArxivDownloader
from dataset_builder.qa_generator import QAGenerator
from dataset_builder.retrieval_simulator import RetrievalSimulator

# ── Reuse production chunking modules ─────────────────────────────────────────
from service.chunking import extract_sections, chunk_sections
from schemas.chunk import Chunk, ChunkMetadata
from prompt.rag_prompt import RAG_SYSTEM_PROMPT

import tiktoken
from uuid import uuid4

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# ── Minimal SectionInfo-like dataclass for compatibility ──────────────────────
from dataclasses import dataclass

@dataclass
class _Section:
    title: str
    text: str
    page_start: int
    page_end: int


def _sections_to_chunks(sections, chunk_size: int, overlap: int) -> list[Chunk]:
    """Convert SectionInfo objects to Chunk objects using production chunk_sections()."""
    raw_docs = chunk_sections(sections, chunk_size, overlap)
    chunks = []
    for i, item in enumerate(raw_docs):
        doc = item["doc"]
        section = item["section"]
        text = doc.page_content
        if not text.strip():
            continue
        metadata = ChunkMetadata(
            paper_id="",   # filled in by caller
            title="",
            section=section.title,
            page_start=section.page_start,
            page_end=section.page_end,
            char_start=doc.metadata.get("start_index"),
            char_end=None,
        )
        chunks.append(
            Chunk(
                chunk_id=str(uuid4()),
                chunk_index=i,
                text=text,
                word_count=len(text.split()),
                token_count=len(encoding.encode(text)),
                metadata=metadata,
            )
        )
    return chunks


# ── DatasetBuilder ─────────────────────────────────────────────────────────────

class DatasetBuilder:
    def __init__(
        self,
        output_path: str = "./data/dataset.jsonl",
        chunk_size: int = 700,
        overlap: int = 150,
        samples_per_paper: int = 4,
    ):
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.samples_per_paper = samples_per_paper

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.downloader = ArxivDownloader(save_dir="./data/arxiv_pdfs")
        self.qa_gen = QAGenerator()
        self.simulator = RetrievalSimulator(self.qa_gen, RAG_SYSTEM_PROMPT)

        # Global pool of chunk texts from already-processed papers (for distractors)
        self.distractor_pool: list[str] = []

        self._stats = Counter()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_kaggle(
        self,
        json_path: str,
        n_papers: int = 10,
        categories: list[str] | None = None,
    ):
        """
        Load paper metadata from the Kaggle arxiv JSONL file,
        download each PDF, and generate samples.
        """
        categories = categories or ["cs.CL", "cs.AI", "cs.LG", "cs.IR"]
        pattern = "|".join(categories)

        processed = 0
        errors = 0
        start = time.time()

        print(f"Starting dataset build | target: {n_papers} papers")
        print(f"Output: {self.output_path}\n")

        for df_chunk in pd.read_json(json_path, lines=True, chunksize=5_000):
            filtered = df_chunk[
                df_chunk["categories"].str.contains(pattern, regex=True, na=False)
            ]

            for _, row in filtered.iterrows():
                if processed >= n_papers:
                    break

                paper_id = str(row["id"])
                title = row["title"]

                print(f"[{processed+1}/{n_papers}] {paper_id} | {title[:70]}")

                try:
                    pdf_path = self.downloader.download(paper_id)
                    if pdf_path is None:
                        errors += 1
                        continue

                    self.process_paper(paper_id, title, pdf_path)
                    processed += 1

                except Exception as e:
                    errors += 1
                    print(f"  [error] {paper_id}: {e}")

            if processed >= n_papers:
                break

        elapsed = time.time() - start
        print(f"\n{'='*50}")
        print(f"Done | {processed} papers | {errors} errors | {elapsed/60:.1f} min")
        self.print_stats()

    def process_paper(self, paper_id: str, title: str, pdf_path: str):
        """Parse PDF → chunk → generate samples → write to JSONL."""
        # 1. Extract sections using production module
        sections = extract_sections(pdf_path)
        if len(sections) < 2:
            print(f"  [skip] {paper_id}: too few sections ({len(sections)})")
            return

        # 2. Chunk using production settings
        chunks = _sections_to_chunks(sections, self.chunk_size, self.overlap)
        for c in chunks:
            c.metadata.paper_id = paper_id
            c.metadata.title = title

        if len(chunks) < 3:
            print(f"  [skip] {paper_id}: too few chunks ({len(chunks)})")
            return

        print(f"  sections={len(sections)} chunks={len(chunks)}")

        # 3. Only add to distractor pool AFTER we have enough (avoid same-paper contamination)
        current_paper_texts = [c.text for c in chunks]

        # 4. Generate samples_per_paper samples
        generated = 0
        for _ in range(self.samples_per_paper * 3):  # extra attempts for failures
            if generated >= self.samples_per_paper:
                break
            if len(self.distractor_pool) < 3:
                # Not enough distractors yet: only EASY samples possible
                sample = self.simulator._easy(paper_id, title, random.choice(chunks))
            else:
                sample = self.simulator.build_sample(
                    paper_id, title, chunks, self.distractor_pool
                )
            if sample:
                self._write(sample)
                self._stats[sample["difficulty"]] += 1
                self._stats[f"type:{sample['question_type']}"] += 1
                generated += 1

        # 5. Add this paper's chunks to the global distractor pool
        self.distractor_pool.extend(current_paper_texts)
        print(f"  generated={generated} | pool_size={len(self.distractor_pool)}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write(self, sample: dict):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def print_stats(self):
        print("\n── Distribution ──────────────────────")
        total = sum(v for k, v in self._stats.items() if not k.startswith("type:"))
        for label in ["EASY", "MEDIUM", "HARD"]:
            count = self._stats.get(label, 0)
            pct = count / total * 100 if total else 0
            print(f"  {label:8s}: {count:4d} ({pct:.1f}%)")
        print("\n── QA Types ──────────────────────────")
        for k, v in self._stats.items():
            if k.startswith("type:"):
                pct = v / total * 100 if total else 0
                print(f"  {k[5:]:20s}: {v:4d} ({pct:.1f}%)")
        print(f"  {'TOTAL':20s}: {total:4d}")
