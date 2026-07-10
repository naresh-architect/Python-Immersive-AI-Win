from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy


# Read the data directory from the environment variable.
# If DATA_DIR is not set, default to the "data" folder.
DATA_DIR = os.getenv("DATA_DIR", "data")


def _load_docs(base: str = DATA_DIR) -> List[Document]:
    """
    Load supported documents from the specified directory and its subdirectories.

    Supported file types:
    - Markdown (.md)
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)

    Each document is tagged with a 'category' based on its parent folder.
    """
    docs: List[Document] = []

    # Recursively search for all files inside the base directory.
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):

        # Skip directories and hidden files.
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue

        # Extract the file extension.
        ext = os.path.splitext(path)[1].lower()

        # Determine the document category from the first folder name.
        # Example:
        # data/python/file1.pdf  -> category = "python"
        # data/java/file2.md     -> category = "java"
        relative_path = os.path.relpath(path, base)
        category = (
            relative_path.split(os.sep)[0]
            if os.sep in relative_path
            else "general"
        )

        try:
            loaded_docs = []

            # Load Markdown documents.
            if ext == ".md":
                for d in UnstructuredMarkdownLoader(path).load():
                    loaded_docs.append(d)

            # Load PDF documents.
            elif ext == ".pdf":
                for d in PyMuPDFLoader(path).load():
                    loaded_docs.append(d)

            # Load Microsoft Word documents.
            elif ext == ".docx":
                for d in UnstructuredWordDocumentLoader(path).load():
                    loaded_docs.append(d)

            # Load plain text documents.
            elif ext == ".txt":
                for d in TextLoader(path).load():
                    loaded_docs.append(d)

            # Attach category metadata to every loaded document.
            for d in loaded_docs:
                d.metadata["category"] = category
                docs.append(d)

        except Exception:
            # Continue processing other files even if one fails.
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    Chunking improves embedding quality and retrieval accuracy.
    """
    splitter = RecursiveCharacterTextSplitter(
        # Maximum characters in each chunk.
        chunk_size=900,

        # Number of overlapping characters between chunks.
        chunk_overlap=120,
    )

    try:
        return splitter.split_documents(docs)

    except Exception:
        print("INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise


async def _create_index(store):
    """
    Create an HNSW vector index to speed up similarity searches.

    HNSW (Hierarchical Navigable Small World) provides efficient
    approximate nearest-neighbor (ANN) search.
    """
    index = HNSWIndex(

        # Name of the database index.
        name="hnsw_idx",

        # Use cosine similarity for comparing embeddings.
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,

        # Number of graph connections per node.
        # Higher value improves recall but increases index size.
        m=16,

        # Controls index construction quality.
        # Higher value creates a better index but takes longer.
        ef_construction=64,
    )

    # Create the index asynchronously.
    await store.aapply_vector_index(index, concurrently=True)

    print("Index created successfully.")


async def run_ingest_async() -> dict:
    """
    Complete ingestion pipeline.

    Steps:
    1. Load documents.
    2. Split documents into chunks.
    3. Connect to the vector store.
    4. Generate embeddings and store them.
    5. Create the HNSW vector index.
    """

    # Load all supported documents.
    docs = _load_docs()

    # Split documents into chunks.
    chunks = _chunk(docs)

    # Connect to PostgreSQL vector store.
    store = await get_vector_store()

    # Generate embeddings and store them in the database.
    await store.aadd_documents(chunks)

    print(f"INGEST: {len(docs)} documents, {len(chunks)} chunks")

    # Build the vector search index.
    await _create_index(store)

    # Return ingestion statistics.
    return {
        "documents": len(docs),
        "chunks": len(chunks),
    }