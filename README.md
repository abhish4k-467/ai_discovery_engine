# Month 1: Foundational RAG - Instructions

## Setup

1. **Install Dependencies**:
   ```bash
   uv add requests beautifulsoup4 qdrant-client pydantic-ai fastembed pypdf2 python-dotenv
   ```

2. **Configuration**:
   Copy `.env.example` to `.env` and fill in your API keys (OpenAI is required for generation).
   Qdrant will default to local storage (`./qdrant_data`) if no endpoint is provided.

   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY
   # Optional for real-time web search fallback: set TAVILY_API_KEY
   ```

3. **Verify Integrations**:
   Start the API and check integration readiness:

   ```bash
   uv run uvicorn src.api:app --reload
   ```

   Then call:

   ```bash
   curl http://127.0.0.1:8000/health
   curl http://127.0.0.1:8000/integrations
   ```

   Expected response includes:
   - `groq_configured: true|false`
   - `tavily_configured: true|false`

### 1. Ingest Documents

**PDF File:**
```bash
python run.py ingest --file path/to/your/document.pdf
```

**Web Page:**
```bash
python run.py ingest --url https://example.com/some-page
```

### 2. Query the Knowledge Base

Ask questions based on the ingested content:

```bash
python run.py query "What is the main topic of the document?"
```

## Structure

- `src/doc_processor.py`: PDF extraction and Web scraping with chunking.
- `src/vector_db.py`: Qdrant vector database interface using FastEmbed.
- `src/rag.py`: Main RAG pipeline orchestrating retrieval and generation.
- `src/main.py`: CLI command handling.
- `run.py`: Entry point script.
