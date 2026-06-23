"""Verify retrieval in ISOLATION (no LLM yet): ask a question, see nearest chunks.

    python demos/retrieval_demo.py
"""

from src.rag import get_vectorstore

store = get_vectorstore()

question = "Who is the lead engineer and what is the budget of Project Zephyr?"
results = store.similarity_search(question, k=2) # nearest 2 chunks

print(f"Question: {question}\n")
for i, doc in enumerate(results, 1):
    print(f"--- nearest chunk {i} ---")
    print(doc.page_content.strip())
    print()
    