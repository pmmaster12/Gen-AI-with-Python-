from mem0 import Memory
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from config import CONFIG
from langsmith import traceable, trace

load_dotenv()

mem_client = Memory.from_config(CONFIG)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2000,
    max_retries=2,
    api_key=os.environ.get("GROQ_API_KEY")
)

@traceable(run_type="retriever", name="memory_search")
def search_memories(message, user_id):
    """Search for relevant memories - this will trace the embedding + vector search"""
    mem_result = mem_client.search(query=message, user_id=user_id)
    results = mem_result.get("results", [])
    memories = "\n".join(m["memory"] for m in results if "memory" in m)
    return memories, results

@traceable(run_type="llm", name="generate_response")
def generate_response(message, memories):
    """Generate response using LLM with memory context"""
    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent.

    Relevant past memory:
    {memories}
    """

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", message)
    ]

    response = llm.invoke(messages)
    return response.content

@traceable(run_type="tool", name="store_memory")
def store_memory(message, answer, user_id):
    """Store interaction in memory - traces the LLM extraction + graph/vector storage"""
    memory_messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    
    result = mem_client.add(memory_messages, user_id=user_id)
    return result

@traceable(run_type="chain", name="memory_chat_pipeline")
def chat(message):
    """Complete chat pipeline with memory"""
    
    # Step 1: Search memories (embedding + vector search)
    memories, search_results = search_memories(message, "p123")
    print(f"Found {len(search_results)} relevant memories")
    print(memories)
    
    # Step 2: Generate response with LLM
    answer = generate_response(message, memories)
    
    # Step 3: Store new memory (LLM extraction + storage)
    storage_result = store_memory(message, answer, "p123")
    
    return answer


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat(user_input)
    print("Assistant:", response)