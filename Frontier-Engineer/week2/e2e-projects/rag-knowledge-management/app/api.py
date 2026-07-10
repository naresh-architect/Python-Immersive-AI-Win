# app/api.py
from __future__ import annotations
import asyncio, time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .rag import answer_with_docs_async
from .ingest import run_ingest_async

# Create the FastAPI application.
# The title appears in the automatically generated Swagger documentation.
app = FastAPI(title="Company Knowledge Assistant")

# ------------------------------------------------------------------
# Serve static frontend files (HTML, CSS, JavaScript, images, etc.)
# ------------------------------------------------------------------
static_dir = Path(__file__).with_name("static")

# Make everything inside the "static" folder available under /static.
# Example:
# http://localhost:8000/static/style.css
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ------------------------------------------------------------------
# Variables used to track document ingestion
# ------------------------------------------------------------------

# Prevent multiple ingestion jobs from running simultaneously.
_ingest_lock = asyncio.Lock()

# Holds the currently running ingestion task.
_ingest_task: asyncio.Task | None = None

# Store the current ingestion status.
_ingest_last = {
    "status": "idle",          # idle | running | succeeded | failed
    "started_at": None,        # Timestamp when ingestion started
    "finished_at": None,       # Timestamp when ingestion completed
    "stats": None,             # Number of documents/chunks processed
    "error": None,             # Error message if ingestion fails
}


# ------------------------------------------------------------------
# Request model for the /ask endpoint
# ------------------------------------------------------------------
class Ask(BaseModel):
    # User's natural language question.
    question: str


# ------------------------------------------------------------------
# Serve the application's main HTML page.
# ------------------------------------------------------------------
@app.get("/")
async def root_page():
    return FileResponse(static_dir / "index.html")


# ------------------------------------------------------------------
# Background ingestion job
# ------------------------------------------------------------------
async def _ingest_job():
    """
    Execute the complete ingestion pipeline in the background.

    This function updates the ingestion status before, during,
    and after processing.
    """

    # Mark ingestion as running.
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None
    })

    try:
        # Load documents, create embeddings,
        # store vectors, and build the vector index.
        stats = await run_ingest_async()

        # Mark ingestion as successful.
        _ingest_last.update({
            "status": "succeeded",
            "finished_at": time.time(),
            "stats": stats
        })

    except Exception as e:
        # Save failure information.
        _ingest_last.update({
            "status": "failed",
            "finished_at": time.time(),
            "error": str(e)
        })


# ------------------------------------------------------------------
# Start ingestion
# ------------------------------------------------------------------
@app.post("/ingest")
async def kick_off_ingest():
    """
    Start document ingestion in the background.

    If another ingestion is already running,
    return HTTP 409 (Conflict).
    """
    global _ingest_task

    async with _ingest_lock:

        # Prevent multiple ingestion jobs from running simultaneously.
        if _ingest_task and not _ingest_task.done():
            return JSONResponse(
                {"ok": False, "message": "Ingestion already running"},
                status_code=409
            )

        # Launch ingestion as a background task.
        _ingest_task = asyncio.create_task(_ingest_job())

    return {
        "ok": True,
        "message": "Ingestion started"
    }


# ------------------------------------------------------------------
# Get ingestion status
# ------------------------------------------------------------------
@app.get("/ingest/status")
async def ingest_status():
    """
    Return the current ingestion status.
    """
    return {
        "ok": True,
        **_ingest_last
    }


# ------------------------------------------------------------------
# Ask a question using the RAG pipeline
# ------------------------------------------------------------------
@app.post("/ask")
async def ask(q: Ask):
    """
    Accept a user question, perform semantic retrieval,
    generate an answer using the LLM, and return the response.
    """

    # Start timer for performance measurement.
    start = time.perf_counter()

    # Optional metadata filter.
    # Only search documents belonging to this category.
    category = "guides"

    # Execute the Retrieval-Augmented Generation pipeline.
    answer, sources, contexts = await answer_with_docs_async(
        q.question,
        category
    )

    # Measure total execution time.
    elapsed = time.perf_counter() - start
    print(f"⏱️ /ask execution took {elapsed:.2f} seconds")

    # Return the generated answer along with retrieved sources.
    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts
    }