from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing_extensions import TypedDict
from groq import Groq
import json
import os
from langsmith import traceable, trace

# ------------------------
# Load env
# ------------------------
load_dotenv()

# IMPORTANT: Make sure these are set in your env
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=xxxx
# LANGCHAIN_PROJECT=groq-langgraph-router

# ------------------------
# Groq client (single instance)
# ------------------------
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ------------------------
# Graph State
# ------------------------
class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool


# ------------------------
# Node: Detect Query Type
# ------------------------
@traceable
def detect_query(state: State):
    with trace(
        name="groq_detect_query_llm",
        run_type="llm",
        inputs={
            "user_message": state["user_message"],
            "model": "openai/gpt-oss-20b",
        },
    ):
        response = groq.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Determine whether the user query is related to programming or coding. "
                        "Return only structured JSON according to the schema."
                    ),
                },
                {"role": "user", "content": state["user_message"]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "detect_query",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_coding_question": {"type": "boolean"}
                        },
                        "required": ["is_coding_question"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    result = json.loads(response.choices[0].message.content)
    state["is_coding_question"] = result["is_coding_question"]
    return state


# ------------------------
# Router
# ------------------------
@traceable
def route_edge(state: State):
    return (
        "solve_coding_question"
        if state["is_coding_question"]
        else "solve_simple_question"
    )


# ------------------------
# Node: Solve Coding Question
# ------------------------
@traceable
def solve_coding_question(state: State):
    with trace(
        name="groq_solve_coding_llm",
        run_type="llm",
        inputs={
            "question": state["user_message"],
            "model": "openai/gpt-oss-120b",
        },
    ):
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert coding assistant. "
                        "Solve the user's programming question clearly and correctly. "
                        "Respond only using the provided JSON schema."
                    ),
                },
                {"role": "user", "content": state["user_message"]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "coding_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"}
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    result = json.loads(response.choices[0].message.content)
    state["ai_message"] = result["answer"]
    return state


# ------------------------
# Node: Solve Simple Question
# ------------------------
@traceable
def solve_simple_question(state: State):
    with trace(
        name="groq_simple_chat_llm",
        run_type="llm",
        inputs={
            "message": state["user_message"],
            "model": "openai/gpt-oss-20b",
        },
    ):
        response = groq.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly conversational assistant. "
                        "Respond naturally and helpfully. "
                        "Respond only using the provided JSON schema."
                    ),
                },
                {"role": "user", "content": state["user_message"]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "simple_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"}
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    result = json.loads(response.choices[0].message.content)
    state["ai_message"] = result["answer"]
    return state


# ------------------------
# Build Graph
# ------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)
graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()


# ------------------------
# Run Graph
# ------------------------
@traceable
def call_graph():
    state: State = {
        "user_message": "hi what is binary search",
        "ai_message": "",
        "is_coding_question": False,
    }

    result = graph.invoke(state)
    print("Final State:", result)
    print("\nAI Response:", result["ai_message"])


if __name__ == "__main__":
    call_graph()
