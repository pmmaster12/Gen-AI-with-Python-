# rag_1_fixed.py

import requests
from langchain_community.document_loaders import PyPDFLoader
# use langchain's text splitter (avoids transformers -> torch)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from api_embeddings import APIEmbeddings

# file path
file_path = r"D:/personal/coharts/gen ai with python/RAG/nodejs.pdf"
loader = PyPDFLoader(file_path)

# load the documents
docs = loader.load()

# initialize text splitter (same chunk_size/overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents=docs)

# embedding model - using API service instead of direct model
# Check if embedding API is running
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("✓ Embedding API is running")
    else:
        raise ConnectionError("Embedding API health check failed")
except requests.exceptions.RequestException:
    raise ConnectionError(
        "Embedding API is not running. Please start it with: python RAG/embedding_server.py"
    )

embedding_model = APIEmbeddings(api_url="http://localhost:8000")

# create Qdrant collection (empty initially)
vector_store = QdrantVectorStore.from_documents(
    documents=[], 
    embedding=embedding_model, 
    collection_name="nodejs-pdf-collection",
    url="http://localhost:6333"
)

# add texts to the vector store
vector_store.add_documents(texts)

# load retriever from existing collection
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs-pdf-collection",
    embedding=embedding_model
)

# search
query = "What is fs module in Node.js?"
results = retriever.similarity_search(query)
print(results)
