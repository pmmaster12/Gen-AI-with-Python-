from mem0 import Memory
import os
from groq import Groq
from dotenv import load_dotenv
from config import CONFIG
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QUADRANT_HOST = os.environ.get("QUADRANT_HOST")

NEO4J_URL= os.environ.get("NEO4J_URL")
NEO4J_USERNAME=os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD= os.environ.get("NEO4J_PASSWORD")


mem_client = Memory.from_config(CONFIG)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def chat(message):
    mem_result = mem_client.search(query=message, user_id="p123")
    results = mem_result.get("results", [])
    memories = "\n".join(m["memory"] for m in results if "memory" in m)
    print(memories)
    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent.

    Relevant past memory:
    {memories}
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1,
        max_tokens=2000
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    # correct memory storage
    mem_client.add(messages, user_id="p123")

    return answer


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat(user_input)
    print("Assistant:", response)
    
# flow - user input -> embedding(user_input) -> vector search(embedding) -> retrieve relevant memories -> LLM with relevant memories -> generate response -> store interaction as new memory
    # Likes pizza with cheese - llm -  user , pizza , cheese 
    
    # cypher quries  - nearbu 5 relation (5 level tak bfs karlo aisa kucn)
    # llm geenrate cypher quires - match p ( user )-[r:LIKES]->(i:Item) return i.name limit 5
    # llm context - ( chunk_text + sub_graph + user_query+ system prompt) -> openai 
     # response
     # response to add memory
     # query translation - 
     # query -  what is machine learning 
     #  what is machine 
     #  what is learning 
     # what is machine learning
     
     # 1,2,3 1,2  2,3
     # cot - chain of thought
     # tree of thoughts 
     
     
     
     
     
    