import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Initialize Langfuse (optional - you'll need to sign up or self-host)
# langfuse_handler = CallbackHandler(
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
#     host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
# )
langfuse_handler = CallbackHandler()

# Connect to your local Ollama
llm = ChatOllama(
    # model="llama3.2:latest",  # or deepseek-r1:latest
    model="deepseek-r1:latest"
    temperature=0.7,
)

# Define the state (context) structure
class AgentState(TypedDict):
    messages: List[str]      # Conversation history
    scratchpad: str          # Working memory
    query: str               # Current user question

# Build a simple graph
builder = StateGraph(AgentState)

# Define nodes
def process_input(state: AgentState) -> AgentState:
    # Add user query to messages
    messages = state.get("messages", [])
    messages.append(state["query"])
    return {"messages": messages}

def generate_response(state: AgentState) -> AgentState:
    # Generate a response using the LLM
    prompt = f"Previous messages: {state['messages']}\nScratchpad: {state.get('scratchpad', '')}\nRespond to the latest query: {state['messages'][-1]}"
    response = llm.invoke(prompt)
    
    # Add response to messages
    messages = state["messages"]
    messages.append(response.content)
    
    return {"messages": messages, "scratchpad": response.content}

# Add nodes to graph
builder.add_node("process_input", process_input)
builder.add_node("generate_response", generate_response)

# Add edges
builder.set_entry_point("process_input")
builder.add_edge("process_input", "generate_response")
builder.add_edge("generate_response", END)

# Compile the graph
app = builder.compile()

# Test it
if __name__ == "__main__":
    result = app.invoke({
        "query": "What is 2+2?",
        "messages": [],
        "scratchpad": ""
    })
    print(result["messages"][-1])