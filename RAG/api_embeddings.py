"""
LangChain-compatible embeddings client for the FastAPI embedding server.
"""
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class APIEmbeddings(Embeddings):
    """Custom embeddings class that calls the FastAPI embedding server."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the API embeddings client.
        
        Args:
            api_url: Base URL of the embedding API server
        """
        self.api_url = api_url.rstrip('/')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling embedding API: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.api_url}/embed/query",
                params={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling embedding API: {e}")
