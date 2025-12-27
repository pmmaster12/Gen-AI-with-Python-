import os
from dotenv import load_dotenv

load_dotenv()

# ========= ENV VARS =========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QUADRANT_HOST", "localhost")

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ========= VALIDATION =========
required_vars = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "NEO4J_URL": NEO4J_URL,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
}

missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

# ========= APP CONFIG =========
CONFIG = {
    "version": "v1.1",

    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-embedding-001"
        },
    },

    "llm": {
        "provider": "groq",
        "config": {
            "api_key": GROQ_API_KEY,
            "model": "openai/gpt-oss-120b",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },

    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": 6333,
            "collection_name": "mem0_gemini_768",
            "embedding_model_dims": 768,
        },
    },

    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
}
