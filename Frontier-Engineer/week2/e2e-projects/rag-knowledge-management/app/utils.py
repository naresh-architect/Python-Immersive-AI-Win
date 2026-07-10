import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

# Load environment variables from .env file (local development).
# Docker deployments set these via env_file in docker-compose.yml.
load_dotenv()

# Read the PostgreSQL connection string from the environment variable.
# Example:
# DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/database_name
PG_CONN_STR = os.getenv("DATABASE_URL")

# Create a PostgreSQL engine using the connection string.
# This engine manages the database connection and is used by PGVector.
PG_ENGINE = PGEngine.from_connection_string(PG_CONN_STR)

# Initialize the OpenAI embedding model.
# This model converts text into high-dimensional vectors that can be stored
# and searched using semantic similarity.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


async def get_vector_store() -> AsyncPGVectorStore:
    """
    Create and return an asynchronous PGVector vector store.

    Configuration:
    - engine: PostgreSQL connection engine.
    - embedding_service: Model used to generate embeddings.
    - table_name: Database table that stores vectors.
    - metadata_json_column: JSON column containing document metadata.
    - metadata_columns: Metadata fields available for filtering searches.
    """
    return await AsyncPGVectorStore.create(
        # PostgreSQL database engine
        engine=PG_ENGINE,

        # Embedding model used to generate vector representations
        embedding_service=embeddings,

        # Table containing vector embeddings
        table_name="langchain_pg_embedding",

        # JSON column storing metadata for each document
        metadata_json_column="langchain_metadata",

        # Metadata fields that can be used in search filters
        metadata_columns=["category"]
    )