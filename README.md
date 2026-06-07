# Research Operating System - ResearchOS

An intelligent, end-to-end scientific research assistant designed to parse, analyze, and extract deep insights from academic papers. ResearchOS acts as an interactive workspace where researchers can upload PDFs, and perform highly contextual Q&A using Large Language Models (LLMs).

## Introduction

### System Architecture

![System Architecture](img/system_architecture.png)

## Overview

A robust, production-grade API backend featuring:
- **FastAPI Core**: High-performance asynchronous API endpoints for document ingestion and contextual Q&A.
- **Advanced Document Processing**: Parses PDFs, analyzes document layouts (YOLO), splits text into semantic chunks, and generates vector embeddings.
- **Retrieval-Augmented Generation (RAG)**: Uses advanced vector search to fetch relevant evidence from papers before generating scientific answers.
- **Contextual Grounding**: API supports receiving specific text snippets to force the AI to answer *only* based on that precise context.

## Project Structure

```text
├── main.py               # FastAPI entry point. Defines all API routes (e.g., /upload, /chat) and sets up the server.
├── config/               # System configurations: loads environment variables and API keys from .env.
├── schemas/              # Pydantic models: defines the strict data structures for API Requests & Responses.
├── service/              # Core Business & RAG Logic:
│   ├── rag.py            # Handles semantic search, context retrieval, and calling the LLM.
│   └── document.py       # Handles PDF parsing, semantic chunking, and generating vector embeddings.
├── prompt/               # Prompt Engineering: stores System Prompts and LLM templates for contextual answering.
├── data/                 # Local Data Storage:
│   ├── chroma/           # Persisted Vector Database storage (contains embedded chunks).
│   └── uploads/          # Temporary storage for raw PDF files uploaded by users.
├── utils/                # Helper utilities: text cleaning, formatting, or error handling functions.
├── pyproject.toml        # Modern Python dependency management and project metadata (used with `uv`).
├── Makefile              # Command shortcuts to run the server, linter, or tests quickly.
└── .env.example          # Template file showing required environment variables (e.g., OPENAI_API_KEY).
```

## Quick Start

### Start the Server

```bash
# Install dependencies and create a virtual environment using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start the server (runs on http://localhost:8000)
make server  # Or: uv run uvicorn main:app --reload
```

## Tech Stack

- **Framework**: Python 3.11+, FastAPI, LangChain
- **Vector DB**: ChromaDB / Qdrant Client (Persistent vector storage)
- **Document Processing**: PyMuPDF (fitz), DocLayout-YOLO (for layout analysis), RecursiveCharacterTextSplitter
- **Embeddings**: OpenAI Embeddings
- **LLMs**: OpenAI GPT-4o / GPT-4o-mini
- **Package Manager**: `uv` (Blazing fast Python dependency manager)

## Key Features

- **Robust RAG Pipeline**: End-to-end extraction, embedding, retrieval, and generation.
- **RESTful API**: Standardized JSON request/response models powered by Pydantic.
- **Modular Architecture**: Clean separation between routes, services, schemas, and data layers.
- **AI-Powered Analysis**: Deep integration with OpenAI for high-quality scientific insights.
