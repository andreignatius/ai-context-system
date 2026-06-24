"""Entry point: build the graph and run a single query.

Run from the langgraph-app/ directory:
    python main.py
"""

from src.graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver


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

def chat():
    # multi-turn chat loop with SQLite checkpointer (memory survives restarts!)
    # checkpointer = MemorySaver() # auto-save store (in RAM for now)
    # app = build_graph(checkpointer)
    db_path = "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer)
        # the "save slot" for this convo
        config = {"configurable": {"thread_id": "compress-test-2"}}

        # # these live across turns / prompts
        # # this is what gives agent its memory
        # messages = []
        # scratchpad = ""

        print("Chat with the agent. Type 'quit' or 'exit' to stop.\n")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            # result = app.invoke(
            #     {
            #         "query": user_input,
            #         "messages": messages, # feed in the entire prior conversation
            #         "scratchpad": scratchpad,
            #     }
            # )

            # note: we only pass the new query, checkpointer reloads the
            # prior messages automatically, keyed by thread_id
            result = app.invoke({"query": user_input}, config)
            reply = result["messages"][-1].content
            print(f"Assistant: {reply}\n")

            # # save the updated state back, so the next turn remembers this one
            # messages = result["messages"]
            # scratchpad = result["scratchpad"]


if __name__ == "__main__":
    # main()
    chat()