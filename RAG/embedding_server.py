from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn

app = FastAPI(title="Embedding Service", description="HuggingFace Embeddings API")

# Initialize embedding model
embedding_model = None

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model on startup"""
    global embedding_model
    try:
        print("Loading HuggingFace embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Test with a sample query
        test_embedding = embedding_model.embed_query("test")
        print(f"✓ Embedding model loaded successfully (dimension: {len(test_embedding)})")
    except Exception as e:
        print(f"✗ Error loading embedding model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Embedding Service is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    return {"status": "healthy", "model": "sentence-transformers/all-MiniLM-L6-v2"}

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for a list of texts
    
    Args:
        request: EmbeddingRequest containing a list of texts
        
    Returns:
        EmbeddingResponse with embeddings for each text
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        # Generate embeddings for all texts
        embeddings = embedding_model.embed_documents(request.texts)
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/query")
async def embed_query(text: str):
    """
    Generate embedding for a single query text
    
    Args:
        text: Single text string to embed
        
    Returns:
        Dictionary with the embedding vector
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        embedding = embedding_model.embed_query(text)
        return {"embedding": embedding, "dimension": len(embedding)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

