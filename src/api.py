import os
import logging
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import RAGSystem

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TheLaunchEngineAPI")

app = FastAPI(
    title="The Launch Engine Consultant API",
    description="Backend RAG system tailored for The Launch Engine's IT and Startup Consultants.",
    version="1.0.0",
)

rag = None


def get_rag():
    global rag
    if rag is None:
        logger.info("Initializing RAGSystem...")
        rag = RAGSystem()
    return rag

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact system admin."},
    )

class URLIngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

class ResearchRequest(BaseModel):
    topic: str


@app.get("/")
def root():
    return {
        "message": "The Launch Engine Consultant API is running",
        "endpoints": [
            "/health",
            "/ingest/file",
            "/ingest/url",
            "/query",
            "/research"
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "integrations": RAGSystem.integration_status(),
    }


@app.get("/integrations")
def integrations():
    """Returns configuration status for external integrations."""
    return RAGSystem.integration_status()


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename:
        logger.warning("Ingest attempt with no file provided.")
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Unsupported file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    temp_path = None

    try:
        logger.info(f"Receiving file {file.filename} for ingestion.")
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        rag_instance = get_rag()
        rag_instance.ingest_file(temp_path)

        logger.info(f"Successfully ingested {file.filename}.")
        return {
            "message": "File ingested successfully",
            "filename": file.filename,
        }

    except Exception as e:
        logger.error(f"Failed to ingest file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ingest/url")
def ingest_url(payload: URLIngestRequest):
    logger.info(f"Receiving URL {payload.url} for ingestion.")
    try:
        rag_instance = get_rag()
        rag_instance.ingest_url(payload.url)

        logger.info(f"Successfully ingested URL {payload.url}.")
        return {
            "message": "URL ingested successfully",
            "url": payload.url,
        }
    except Exception as e:
        logger.error(f"Failed to ingest URL {payload.url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")


@app.post("/query")
def query(payload: QueryRequest):
    logger.info(f"Processing query: {payload.question}")
    try:
        rag_instance = get_rag()
        answer = rag_instance.query(payload.question)
        
        logger.info("Successfully generated query response.")
        return {
            "question": payload.question,
            "answer": answer,
        }

    except Exception as e:
        logger.error(f"Query generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/research")
def research(payload: ResearchRequest):
    """Generates a deep-dive research report structured for consultants."""
    logger.info(f"Triggering deep research task for topic: {payload.topic}")
    try:
        rag_instance = get_rag()
        report = rag_instance.generate_research_report(payload.topic)
        
        logger.info("Successfully generated research report.")
        return {
            "topic": payload.topic,
            "report": report,
        }

    except Exception as e:
        logger.error(f"Research task failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research task failed: {str(e)}")
