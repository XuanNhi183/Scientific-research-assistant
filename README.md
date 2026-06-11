# Research Operating System - ResearchOS

An intelligent, end-to-end scientific research assistant designed to parse, analyze, and extract deep insights from academic papers. ResearchOS acts as an interactive workspace where researchers can upload PDFs and perform highly contextual Q&A using Large Language Models (LLMs).

---

## System Architecture

The system follows a **Modular** design with four main components:

- **Frontend (React, Vite, TailwindCSS)**: Interactive UI with dynamic rendering based on question language, integrated PDF Viewer and Chat Assistant.
- **Backend (FastAPI)**: API Server handling business logic via `main.py`, `schemas/` (Pydantic validation), and `service/` (core RAG/LLM/Chunking logic).
- **Data & Storage**: `data/uploads/` for raw PDFs, **ChromaDB** as the persistent vector store.
- **AI Models**:
  - `text-embedding-3-small` (OpenAI) — vector embeddings
  - `gpt-4o-mini` (OpenAI) — JSON extraction, paper analysis, and dataset generation
  - `Qwen2.5-7B-Instruct` — local LLM via Ollama for RAG Q&A

![System Architecture](img/system_architecture.png)

---

## Project Structure

```text
├── frontend/                  # React & Vite frontend application
├── main.py                    # FastAPI entry point: defines all API routes (/upload, /chat, etc.)
├── config/
│   └── dataset_config.yaml    # Parameters for dataset generation (n_papers, categories, etc.)
├── schemas/
│   ├── chunk.py               # Chunk & ChunkMetadata models
│   ├── document.py            # Document upload/response models
│   ├── paper.py               # Paper metadata models
│   ├── rag.py                 # RAG query/response models (QuestionRequest, AnswerResponse, AnalyzeRequest)
│   └── section.py             # Section extraction models
├── service/
│   ├── rag_service.py         # Orchestrates the full RAG pipeline (retrieve → generate)
│   ├── document_service.py    # PDF ingestion: parsing & storing
│   ├── chunking.py            # Section extraction & text splitting logic
│   ├── embedding_service.py   # OpenAI embedding calls
│   ├── chroma_service.py      # ChromaDB operations (upsert, query, delete)
│   └── llm_service.py         # LLM API calls & prompt formatting
├── prompt/
│   └── rag_prompt.py          # System prompt template for contextual Q&A
├── dataset_builder/           # Offline pipeline for generating SFT fine-tuning dataset
│   ├── build_dataset.py       # Entry point: reads config/dataset_config.yaml → runs pipeline
│   ├── dataset_builder.py     # Orchestrator: download → chunk → generate → write JSONL
│   ├── qa_generator.py        # Generates reasoning-heavy Q&A pairs via OpenAI
│   ├── retrieval_simulator.py # Simulates RAG retrieval difficulty (EASY/MEDIUM/HARD)
│   └── arxiv_downloader.py    # Downloads PDFs from arXiv by paper ID
├── utils/
│   └── processing_pdf.py      # PDF pre-processing helpers
├── data/
│   ├── chroma/                # Persisted ChromaDB vector store
│   ├── uploads/               # Temporary storage for uploaded PDFs
│   └── dataset.jsonl          # Generated SFT dataset
├── docs/                      # Technical documentation and guides
├── notebook/                  # Jupyter notebooks for QLoRA fine-tuning and evaluation
├── scripts/                   # Debugging and utility scripts
├── Modelfile                  # Ollama Modelfile for loading the fine-tuned GGUF model
├── pyproject.toml             # Python dependency management (used with `uv`)
├── Makefile                   # Shortcuts: `make run`, `make dataset`, `make dataset-config`
└── .env.example               # Template for required environment variables
```

---

## Quick Start

### 1. Environment Variables

Copy the example env file and populate your API keys:

```bash
cp .env.example .env
```

Set `OLLAMA_BASE_URL` in `.env`:
- Running Ollama locally: `http://localhost:11434/v1`
- Running Ollama on Google Colab via Ngrok: `https://xxxx.ngrok-free.app/v1`

### 2. Start the API Server

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Start the FastAPI server (runs on http://localhost:8000)
make run  # Or: PYTHONPATH=. uv run python main.py
```

### 3. Build the Fine-Tuning Dataset

Edit [`config/dataset_config.yaml`](config/dataset_config.yaml) to configure `n_papers`, `categories`, and output path, then run:

```bash
make dataset

# To use a custom config file:
make dataset-config CONFIG=config/my_config.yaml
```

---

## Production Pipeline

### Document Ingestion

When a user uploads a PDF, the following steps run through `document_service.py` and `chunking.py`:

1. **Storage**: PDF saved to `data/uploads/` with a generated UUID (`file_id`).
2. **Text Extraction**: PyMuPDF (`fitz`) reads each page and extracts raw text.
3. **Section Identification**: Headings detected via font size and bold formatting. Noisy sections (`References`, `Acknowledgements`, `Appendix`, `Declarations`) are automatically filtered.
4. **Semantic Chunking**: `RecursiveCharacterTextSplitter` splits text (`chunk_size=700`, `overlap=150`). Chunks that are too short or contain excessive noise characters are discarded.
5. **Embedding & Storage**: Each chunk is vectorized via `text-embedding-3-small` and stored in ChromaDB alongside metadata (`paper_id`, `section`, `page`, `chunk_index`).
6. **Full Paper Analysis**: `gpt-4o-mini` automatically extracts a structured JSON summary including Abstract, Metrics, Key Findings, and Glossary on upload.

### Retrieval-Augmented Generation (RAG)

When a user asks a question, `rag_service.py` and `llm_service.py` execute:

```text
User Question
  → Embed question (text-embedding-3-small)
  → Vector search ChromaDB (Top-5 chunks, filtered by paper_id)
  → Context ordering: Chunk 0 (Title/Authors) forced to position #1 to prevent "Lost in the Middle"
  → Detect question language (Vietnamese / English) → instruct LLM to reply in same language
  → Qwen2.5-7B-Instruct generates answer (returns INSUFFICIENT_INFORMATION if context is missing)
  → API returns answer + source list (page references)
  → React Frontend renders source UI in matching language
```

---

## Offline Pipeline: Dataset Builder

The fine-tuning dataset pipeline lives in `dataset_builder/`, configured via `config/dataset_config.yaml`:

1. **Data Sourcing**: Reads `arxiv-metadata.json` from the [Cornell University arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) (Kaggle), filters by category (`cs.AI`, `cs.CL`), and auto-downloads PDFs via `ArxivDownloader`.
2. **Chunking**: Reuses the same Production chunking module to ensure the model trains on real-world format.
3. **Reasoning-Heavy QA Generation** (`qa_generator.py`): Forces comparative, trade-off, and cross-section questions — extractive/copy-paste questions are explicitly banned.
4. **Retrieval Simulation** (`retrieval_simulator.py`): Distributes context difficulty:
   - **35% EASY**: 1 highly relevant chunk.
   - **40% MEDIUM**: 2 chunks with split information, or 1 correct + 1 distractor.
   - **25% HARD**: Multiple noisy chunks, or no relevant information (teaches the model to refuse and return `INSUFFICIENT_INFORMATION`).
5. **LLM Validator**: `gpt-4o-mini` reviews all generated pairs and removes hallucinated or trivially simple questions.
6. **Output**: JSONL file in ChatML format (`system`, `user`, `assistant`).

---

## Fine-Tuning (Supervised Fine-Tuning with QLoRA)

Training is done with **Unsloth** and **TRL SFTTrainer** on Google Colab (GPU T4), using QLoRA 4-bit quantization.

| Parameter | Value |
|---|---|
| Base Model | `unsloth/Qwen2.5-7B-Instruct` |
| Dataset | `xunnhi/QA-Dataset-Generator` (~700 samples, ChatML format) |
| Max Sequence Length | `2048` |
| Fine-Tuned Model | [`xunnhi/Qwen2.5-7B-RAG-LoRA`](https://huggingface.co/xunnhi/Qwen2.5-7B-RAG-LoRA) |
### Deploying the Fine-Tuned Model

After training, load the GGUF file into Ollama:

```bash
ollama create <model_name> -f Modelfile
```

Then update the `OLLAMA_BASE_URL` in `.env` to point the backend at the new model.

---

## Evaluation Results (50 Scenarios)

| Model | Faithfulness | Relevance | Refusal Accuracy |
|---|---|---|---|
| Fine-tuned Qwen 1.5B | 3.60 / 5.0 | 3.62 / 5.0 | 100% (10/10) |

- **Faithfulness**: How closely the model sticks to provided context without hallucinating (0.0–5.0).
- **Relevance**: How directly the model addresses the specific query (0.0–5.0).
- **Refusal Accuracy**: Correctly returning `INSUFFICIENT_INFORMATION` when context is absent.

> The fine-tuned model achieves perfect refusal accuracy — it never hallucinates when context is missing. Comparison against OpenAI API baselines is pending.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Python 3.11+, FastAPI, LangChain |
| Vector DB | ChromaDB (local persistent) |
| Document Processing | PyMuPDF (fitz), RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLMs | OpenAI `gpt-4o-mini`, Qwen2.5-7B-Instruct (Ollama) |
| Fine-Tuning | Unsloth, TRL SFTTrainer, QLoRA |
| Package Manager | `uv` |
