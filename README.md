# Distributed OCR + Agentic RAG

A document intelligence system that combines Optical Character Recognition (OCR), distributed document processing, and Agentic Retrieval-Augmented Generation (RAG) to extract information from documents and answer questions using relevant retrieved context.

The project aims to support document ingestion, text extraction, indexing, and context-aware question answering through a modular AI pipeline.

## Overview

Documents often contain a mixture of digital text, scanned pages, tables, and image-based content. Extracting and searching this information manually can be inefficient.

This project integrates OCR and RAG into a document-processing workflow. OCR extracts text from scanned or image-based documents, while retrieval and language models help users obtain answers from the extracted content.

Distributed processing components are used to organize document-processing tasks independently from the question-answering workflow.

## Key Features

- **OCR-Based Text Extraction:** Extract text from scanned documents and image-based content.
- **Multiple Document Inputs:** Process supported document and image formats through the ingestion pipeline.
- **Document Processing Pipeline:** Extract, clean, and prepare text for downstream indexing.
- **Distributed Task Processing:** Use Celery workers and Redis for background task execution and queue management.
- **Object Storage:** Use MinIO to store document files and related objects.
- **Metadata Management:** Use PostgreSQL for structured document metadata, where configured.
- **Hybrid Retrieval:** Combine semantic vector search and BM25 keyword retrieval.
- **Reciprocal Rank Fusion:** Combine retrieval rankings to improve candidate selection.
- **Cross-Encoder Reranking:** Reorder retrieved passages based on relevance, where enabled.
- **Agentic RAG:** Use LangGraph-based workflow orchestration to support context retrieval and question answering.
- **Vector Database:** Use Milvus to store and retrieve document embeddings.
- **API Integration:** Expose application functionality through FastAPI endpoints.
- **Containerized Services:** Use Docker to simplify local service setup, where configured.

## System Architecture

```text
              User
               |
               v
       Document Upload API
               |
               v
       Background Task Queue
             Redis
               |
               v
        Celery Workers
               |
               v
       Document Processing
               |
       +-------+--------+
       |                |
       v                v
 Digital Text         OCR Engine
 Extraction           Scanned Content
       |                |
       +-------+--------+
               |
               v
        Text Cleaning
        and Chunking
               |
               v
           Embeddings
               |
               v
          Milvus Store
               |
               v
          User Query
               |
               v
       Retrieval Pipeline
       +-------+--------+
       |                |
       v                v
 Semantic Search     BM25 Search
       |                |
       +-------+--------+
               |
               v
          RRF Fusion
               |
               v
       Cross-Encoder Reranking
               |
               v
        Agentic RAG Workflow
               |
               v
        Contextual Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| OCR | Tesseract OCR, PaddleOCR |
| Image Processing | OpenCV |
| Background Tasks | Celery |
| Message Broker / Queue | Redis |
| Object Storage | MinIO |
| Metadata Database | PostgreSQL |
| Vector Database | Milvus |
| Keyword Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Agent Workflow | LangGraph |
| Containerization | Docker |

## Project Structure

```text
distributed_ocr_agentic_rag/
├── app/
│   ├── api/
│   ├── ocr/
│   ├── ingestion/
│   ├── retrieval/
│   └── agents/
├── tests/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

*Update this tree to match the actual repository directories and filenames.*

## How It Works

### 1. Document Upload
The user submits a supported document through the API or application interface.

### 2. Background Processing
Document-processing jobs are placed in a queue and handled by background workers, where the distributed task-processing components are configured.

### 3. Text Extraction
Digitally extractable text is processed directly. OCR is used to recognize text from scanned pages or image-based documents.

### 4. Text Preparation
Extracted text is cleaned, divided into manageable chunks, and prepared for embedding generation.

### 5. Document Indexing
Embeddings and associated document information are indexed for retrieval.

### 6. Hybrid Retrieval
A user submits a question. Semantic search finds conceptually relevant passages, while BM25 identifies keyword-based matches.

### 7. Fusion and Reranking
RRF combines candidate rankings. A Cross-Encoder can rerank the retrieved passages to prioritize relevant context.

### 8. Agentic Question Answering
The agent workflow uses the retrieved context to produce an answer grounded in available document information.

## Prerequisites

- Python version compatible with the project dependencies
- Docker Desktop, if using containerized services
- Redis
- Milvus
- MinIO and PostgreSQL, if enabled in the configuration
- OCR engines required by the implementation
- Required LLM credentials, if the configured model uses an external API

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/grarun2001/distributed_ocr_agentic_rag.git
cd distributed_ocr_agentic_rag
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file based on `.env.example`, if provided.

Add the required service URLs, database configuration, and model credentials according to the project's configuration.

Never commit API keys, passwords, or other secrets to GitHub.

### 5. Start Required Services

Start the configured infrastructure services, such as Redis, Milvus, MinIO, and PostgreSQL.

If the repository provides a Docker Compose configuration, use the service startup instructions defined for that file.

### 6. Start the Application

Start the FastAPI application and Celery worker using the actual module paths and configuration defined in the repository.

Refer to the application entry points for the exact commands and available endpoints.

## Testing

Run the project's tests, if configured:

```bash
python -m pytest -v
```

Test with different supported input types, including digitally generated PDFs, scanned documents, and images. Validate OCR output and verify that retrieved context supports the generated answers.

## Potential Use Cases

- Searching scanned reports and document collections
- Extracting text from invoices and receipts
- Processing image-based PDFs
- Building searchable document repositories
- Asking questions about indexed documents
- Automating document ingestion and information retrieval

## Future Improvements

- Improve OCR accuracy for handwriting, tables, and complex layouts.
- Add OCR confidence scoring and extraction-quality checks.
- Evaluate retrieval quality using precision, recall, and ranking metrics.
- Add robust retries, task monitoring, and failure recovery.
- Improve document-level filtering and source traceability.
- Add performance benchmarks for processing time and retrieval latency.

## Author

**Arun GR**

GitHub: [grarun2001](https://github.com/grarun2001)

---

*This project explores the integration of OCR, background processing, hybrid information retrieval, and agent-based question answering for document intelligence.*
