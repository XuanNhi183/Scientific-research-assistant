# Research Operating System - ResearchOS

An intelligent, end-to-end scientific research assistant designed to parse, analyze, and extract deep insights from academic papers. ResearchOS acts as an interactive workspace where researchers can upload PDFs, and perform highly contextual Q&A using Large Language Models (LLMs).

## Introduction

### System Architecture

![System Architecture](img/system_architecture.png)

## Overview

A robust, production-grade API backend featuring:
- **FastAPI Core**: High-performance asynchronous API endpoints for document ingestion and contextual Q&A.
- **Advanced Document Processing**: Parses PDFs using PyMuPDF, splits text into semantic chunks, and generates vector embeddings.
- **Retrieval-Augmented Generation (RAG)**: Uses advanced vector search to fetch relevant evidence from papers before generating scientific answers.
- **Contextual Grounding**: API supports receiving specific text snippets to force the AI to answer *only* based on that precise context.

## Project Structure

```text
├── main.py                    # FastAPI entry point: defines all API routes (/upload, /chat, etc.)
├── config/                    # Configuration files
│   └── dataset_config.yaml    # Parameters for dataset generation (n_papers, categories, etc.)
├── schemas/                   # Pydantic models: strict data structures for API Requests & Responses
│   ├── chunk.py               # Chunk & ChunkMetadata models
│   ├── document.py            # Document upload/response models
│   ├── paper.py               # Paper metadata models
│   ├── rag.py                 # RAG query/response models (QuestionRequest, AnswerResponse, AnalyzeRequest)
│   └── section.py             # Section extraction models
├── service/                   # Core Business & RAG Logic
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
│   ├── qa_generator.py        # Generates questions & answers via OpenAI
│   ├── retrieval_simulator.py # Simulates RAG retrieval scenarios (EASY/MEDIUM/HARD)
│   └── arxiv_downloader.py    # Downloads PDFs from arXiv by paper ID
├── utils/
│   └── processing_pdf.py      # PDF pre-processing helpers
├── data/                      # Local data storage (git-ignored)
│   ├── chroma/                # Persisted ChromaDB vector store
│   ├── uploads/               # Temporary storage for uploaded PDFs
│   └── dataset.jsonl          # Generated SFT dataset
├── pyproject.toml             # Python dependency management (used with `uv`)
├── Makefile                   # Shortcuts: `make run`, `make dataset`, `make dataset-config`, etc.
└── .env.example               # Template for required environment variables
```

## Quick Start

### Start the API Server

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start the FastAPI server (runs on http://localhost:8000)
make run  # Or: PYTHONPATH=. uv run python main.py
```

### Build the Fine-Tuning Dataset

Edit [`config/dataset_config.yaml`](config/dataset_config.yaml) to configure number of papers, categories, and output path, then run:

```bash
make dataset

# To use a custom config file:
make dataset-config CONFIG=config/my_config.yaml
```

## Tech Stack

- **Framework**: Python 3.11+, FastAPI, LangChain
- **Vector DB**: ChromaDB (Local persistent vector store)
- **Document Processing**: PyMuPDF (fitz) for text & page extraction, RecursiveCharacterTextSplitter for semantic chunking
- **Embeddings**: OpenAI Embeddings (`text-embedding-3-small`)
- **LLMs**: OpenAI `gpt-4o-mini` for (Q&A), (paper analysis), and (dataset generation)
- **Package Manager**: `uv`

## Key Features

- **Robust RAG Pipeline**: End-to-end extraction, embedding, retrieval, and generation.
- **RESTful API**: Standardized JSON request/response models powered by Pydantic.
- **Modular Architecture**: Clean separation between routes, services, schemas, and data layers.
- **AI-Powered Analysis**: Deep integration with OpenAI for high-quality scientific insights.
