"""One-off: ingest the sample document into Chroma. Run ONCE.

    python ingest.py
"""

from src.rag import ingest

if __name__ == "__main__":
    ingest("data/facts.md")
