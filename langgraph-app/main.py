"""Entry point: build the graph and run a single query.

Run from the langgraph-app/ directory:
    python main.py
"""

from src.graph import build_graph


def main():
    app = build_graph()
    result = app.invoke(
        {
            "query": "What is 2+2?",
            "messages": [],
            "scratchpad": "",
        }
    )
    # messages[-1] is the AIMessage; .content is the text.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
