"""RAG (the Select pillar): embed documents into Chroma, retrieve relevant chunks.

INGEST: read doc -> split into chunks -> embed -> store in Chroma (on disk).
QUERY:  embed a question -> return the k nearest chunks.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# where Chroma persists its vectors on disk
# (local folder, like checkpoints.sqlite)
CHROMA_DIR = "chroma_db"

def get_embeddings():
    """The embedding model, served by Ollama (same engine as chat model)
    
    must be the same model for ingest and query 
    otherwise vectors aren't comparable (lesson 006)
    """
    return OllamaEmbeddings(model="nomic-embed-text")

def get_vectorstore():
    # open (or create) the Chroma vector store on disk
    return Chroma(
        collection_name="docs",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

def ingest(path: str) -> int:
    # read text file, split into chunks, embed them, store in Chroma
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # 300 chars per chunk
        chunk_overlap=50, # overlap so an idea spanning a boundary survives
    )
    chunks = splitter.split_text(text)
    
    store=get_vectorstore()
    store.add_texts(chunks)
    print(f"Ingested {len(chunks)} chunks from {path} into '{CHROMA_DIR}/'.")
    
    return len(chunks)