import os
import json
import re
import time
import random
from collections import Counter

import pandas as pd

from dataset_builder.arxiv_downloader import ArxivDownloader
from dataset_builder.qa_generator import QAGenerator
from dataset_builder.retrieval_simulator import RetrievalSimulator

from service.chunking import extract_sections, chunk_sections, init_marker_models, extract_with_marker, chunk_markdown
from schemas.chunk import Chunk, ChunkMetadata
from prompt.rag_prompt import RAG_SYSTEM_PROMPT

import tiktoken
from uuid import uuid4
from service.chunking import is_data_chunk

encoding = tiktoken.encoding_for_model("text-embedding-3-small")

# ── Minimal SectionInfo-like dataclass for compatibility ──────────────────────
from dataclasses import dataclass

@dataclass
class _Section:
    title: str
    text: str
    page_start: int
    page_end: int



_SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgements",
    "acknowledgments", "appendix", "about the author",
    "funding", "conflict of interest", "declarations",
    "author contributions","reference","references"
}

def _is_valid_section(section) -> bool:
    """Return False for noisy sections (References, Ack, etc.) or near-empty ones."""
    title_lower = (section.title or "").lower().strip()
    # Remove known noise section titles
    if any(skip in title_lower for skip in _SKIP_SECTIONS):
        return False
    # Remove sections whose text is too short to contain meaningful content
    if len(section.text.strip()) < 200:
        return False
        
    # Content-based filter: skip section if >25% lines are citations
    lines = [l.strip() for l in section.text.splitlines() if l.strip()]
    if len(lines) > 5:
        bracket_refs = sum(1 for l in lines if re.match(r'^\s*\[\d+\]', l))
        author_year_lines = sum(1 for l in lines if re.search(r'\(\d{4}\)', l))
        venue_lines = sum(1 for l in lines if re.search(r'\b(In Proceedings|Journal of|arXiv preprint|vol\.|pp\.)\b', l, re.IGNORECASE))
        
        if (bracket_refs + author_year_lines + venue_lines) / len(lines) > 0.25:
            return False

    return True


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

        # Filter 1: chunk too short to contain meaningful content
        if len(text.strip()) < 300:
            continue
            
        # Filter 1.5: data chunk (second defensive layer)
        if is_data_chunk(text):
            continue

        # Filter 2: chunk looks like a reference list
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 2:
            # Type A: Bracket references (e.g. "[1] Li Dong...")
            bracket_refs = sum(1 for l in lines if re.match(r'^\s*\[\d+\]', l))
            if bracket_refs >= 1:
                continue
                
            # Type B: Author/Title/Venue references (APA, Harvard, etc.)
            author_year_lines = sum(1 for l in lines if re.search(r'\(\d{4}\)', l))
            venue_lines = sum(1 for l in lines if re.search(r'\b(In Proceedings|Journal of|arXiv preprint|vol\.|pp\.)\b', l, re.IGNORECASE))
            
            # Lower threshold: if >20% of lines look like citations, drop it
            if (author_year_lines + venue_lines) / len(lines) > 0.2:
                continue
                
        section_title = section.title if section.title else "Unknown"
        metadata = ChunkMetadata(
            paper_id="",   # filled in by caller
            title="",
            section=section_title,
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
        # Lưu ý: Thêm dấu ngoặc tròn () để gọi hàm
        self.marker_models = init_marker_models()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.downloader = ArxivDownloader(save_dir="./data/arxiv_pdfs")
        self.qa_gen = QAGenerator()
        self.simulator = RetrievalSimulator(self.qa_gen, RAG_SYSTEM_PROMPT)

        # Global pool of chunks from already-processed papers (for distractors)
        self.distractor_pool: list[Chunk] = []

        self._stats = Counter()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_kaggle(
        self,
        json_path: str,
        n_papers: int,
        categories: list[str] | None = None,
        min_year: int = 2018,
    ):
        categories = categories or ["cs.CL", "cs.AI", "cs.LG", "cs.IR"]
        pattern = "|".join(categories)

        processed = 0
        errors = 0
        skipped_old = 0
        start = time.time()

        print(f"Starting dataset build | target: {n_papers} papers | min_year: {min_year}")
        print(f"Output: {self.output_path}\n")

        # Clear existing file to prevent appending old generated samples
        open(self.output_path, "w", encoding="utf-8").close()

        for df_chunk in pd.read_json(json_path, lines=True, chunksize=5_000):
            filtered = df_chunk[
                df_chunk["categories"].str.contains(pattern, regex=True, na=False)
            ]

            for _, row in filtered.iterrows():
                if processed >= n_papers:
                    break

                paper_id = str(row["id"])
                normalized_id = ArxivDownloader.normalize_id(paper_id)

                # Skip old-format IDs (e.g. "cs/0704.0047") — pre-2007 papers
                if "/" in normalized_id:
                    skipped_old += 1
                    continue

                # Extract year directly from normalized_id (e.g. "1804.1234" -> 2018, "0706.0022" -> 2007)
                try:
                    paper_year = 2000 + int(normalized_id[:2])
                except (ValueError, IndexError):
                    # Fallback to update_date if format is unexpected
                    update_date = str(row.get("update_date", ""))
                    paper_year = int(update_date[:4]) if update_date else 2000

                if paper_year < min_year:
                    skipped_old += 1
                    continue

                title = row["title"]
                print(f"[{processed+1}/{n_papers}] {paper_id} | {title[:70]}")

                try:
                    pdf_path = self.downloader.download(paper_id)
                    if pdf_path is None:
                        errors += 1
                        continue

                    category = row.get("categories", "").split(" ")[0] if "categories" in row else ""
                    self.process_paper(paper_id, title, pdf_path, category)
                    processed += 1

                except Exception as e:
                    errors += 1
                    print(f"  [error] {paper_id}: {e}")

            if processed >= n_papers:
                break

        elapsed = time.time() - start
        print(f"\n{'='*50}")
        print(f"Done | {processed} papers | {errors} errors | {skipped_old} skipped (old) | {elapsed/60:.1f} min")

        self.print_stats()

    def process_paper(self, paper_id: str, title: str, pdf_path: str, category: str = ""):
        """Parse PDF → chunk → generate samples → write to JSONL."""
        # 1. Extract sections using production module
        chunks = []
        sections = None
        try:
            print(f"  [extractor] Using Marker for {pdf_path} ...")
            md_text = extract_with_marker(pdf_path, self.marker_models)
            chunks = chunk_markdown(md_text)
            print("  [extractor] Marker processing was successful!")
        except Exception as e:
            print(f"  [extractor] Marker error ({e})")
            sections = extract_sections(pdf_path)

            # Filter out noisy sections (References, Acknowledgements, etc.)
            before = len(sections)
            sections = [s for s in sections if _is_valid_section(s)]
            filtered = before - len(sections)
            if filtered:
                print(f"  [filter] dropped {filtered} noisy sections (References/Ack/etc.)")

            if len(sections) < 2:
                print(f"  [skip] {paper_id}: too few sections after filtering ({len(sections)})")
                return

            # 2. Chunk using production settings
            chunks = _sections_to_chunks(sections, self.chunk_size, self.overlap)
        
        # Điền metadata chung cho tất cả các chunk (dù từ Marker hay PyMuPDF)
        for c in chunks:
            c.metadata.paper_id = paper_id
            c.metadata.title = title
            c.metadata.category = category

        # Need at least 5 chunks: guarantees enough variety for EASY/MEDIUM/HARD sampling
        # and avoids wasting LLM calls on papers that are mostly data/references after filtering.
        if len(chunks) < 5:
            print(f"  [skip] {paper_id}: too few chunks after filtering ({len(chunks)}) — skipping")
            return

        num_sections = len(sections) if sections is not None else "N/A (marker)"
        print(f"  sections={num_sections} chunks={len(chunks)}")

        # 3. Chunks for this paper
        current_paper_chunks = chunks

        # 4. Generate samples_per_paper samples
        generated = 0
        valid_samples_buffer = []
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
                valid_samples_buffer.append(sample)
                generated += 1

        MIN_VALID_RATIO = 0.25
        valid_ratio = generated / self.samples_per_paper if self.samples_per_paper > 0 else 0
        if valid_ratio < MIN_VALID_RATIO:
            print(f"  [skip] {paper_id}: weak paper, only {generated}/{self.samples_per_paper} valid samples ({valid_ratio*100:.1f}%) < 25% — dropping.")
            return

        for sample in valid_samples_buffer:
            self._write(sample)
            self._stats[sample["difficulty"]] += 1
            self._stats[f"type:{sample['question_type']}"] += 1

        # 5. Add this paper's chunks to the global distractor pool
        self.distractor_pool.extend(chunks)
        print(f"  generated={generated}/{self.samples_per_paper} | pool_size={len(self.distractor_pool)}")

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