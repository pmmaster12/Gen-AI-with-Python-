# 🚀 GenAI Cohort – Lecture Notes, Code & Resources

This repository contains **lecture-wise notes, code examples, research papers, and tooling references** from a **Generative AI Cohort**, covering everything from **GenAI fundamentals** to **Advanced RAG, Graph Databases, LangGraph, and AI App Tracing**.

---

## 📌 Table of Contents
- [Lecture 1: Overview of Generative AI](#lecture-1-overview-of-generative-ai)
- [Lecture 2: Prompt Engineering](#lecture-2-prompt-engineering)
- [Lecture 3: AI Agents & Tool Calling](#lecture-3-ai-agents--tool-calling)
- [Lecture 4: Introduction to RAG](#lecture-4-introduction-to-rag)
- [Lecture 5: Advanced RAG – Query Translation](#lecture-5-advanced-rag--query-translation)
- [Lecture 6: Graph Databases](#lecture-6-graph-databases)
- [Lecture 7: Tracing AI Applications](#lecture-7-tracing-ai-applications)
- [Lecture 8: LangGraph & Agent Routing](#lecture-8-langgraph--agent-routing)

---

## Lecture 1: Overview of Generative AI

### Topics Covered
- What is Generative AI
- LLM fundamentals
- Embeddings & tokenization

### Notes
- Class Notes: https://app.eraser.io/workspace/WVuECSpptFUySQegvL2b

### Tools & Visualizers
- Vector Embedding Visualizer: https://projector.tensorflow.org/
- Tokenizer Visualizer: https://tiktokenizer.vercel.app

---

## Lecture 2: Prompt Engineering

### Topics Covered
- Prompt design fundamentals
- Zero-shot & Few-shot prompting
- Prompt structuring techniques

### Notes
- Notes: https://app.eraser.io/workspace/7gwrRuUm9lCAugulZEtL

---

## Lecture 3: AI Agents & Tool Calling

### Topics Covered
- AI agents architecture
- Tool calling & function calling
- Ollama
- Hugging Face Transformers

### Notes
- Notes: https://app.eraser.io/workspace/xaA8WBpMpYV5Q3EMPEjT

### Code (Google Colab)
- Colab Notebook 1:  
  https://colab.research.google.com/drive/15rfchi-okbLLNIqmi1RUMqQ8GorWxtgu?usp=sharing
- Colab Notebook 2:  
  https://colab.research.google.com/drive/1C9yJuXSlcetvdZJtQU5vGSeIu8Tq5BTh?usp=sharing

---

## Lecture 4: Introduction to RAG

### Topics Covered
- What is RAG
- Vector databases
- Basic RAG pipeline

### Notes
- Notes: https://app.eraser.io/workspace/zjUq4ETunws45jthYzRh

### Code & Configuration
- Docker Compose (Vector DB):  
  https://github.com/piyushgarg-dev/genai-cohort/blob/main/docker-compose.db.yml

---

## Lecture 5: Advanced RAG – Query Translation

### Topics Covered
- Multi-Query Retrieval
- Query Decomposition
- Chain-of-Thought (CoT)
- Step-Back Prompting
- HyDE

### Notes
- Notes: https://app.eraser.io/workspace/W1ItJUWco1xYkXrVLsMR

### Research Paper
- RAG Query Translation Paper:  
  https://arxiv.org/abs/2310.06117

---

## Lecture 6: Graph Databases

### Topics Covered
- Graph DB fundamentals
- Knowledge graphs
- Graph DB vs Vector DB

### Notes
- Notes: https://app.eraser.io/workspace/OztkHHJmaLO9E2mj005h

### Code Repository
- Graph DB Examples:  
  https://github.com/pmmaster12/Gen-AI-with-Python-/tree/master/graph_db

---

## Lecture 7: Tracing AI Applications

### Topics Covered
- AI app observability
- Tracing LLM calls
- Contextual retrieval
- Langfuse integration

### Notes
- Notes: https://app.eraser.io/workspace/WLej9G0tAs1DoNYaGc7W

### Code
- Weather Agent Example:  
  https://github.com/piyushgarg-dev/genai-cohort/blob/main/weather_agent.py

### Docker Configuration
- Langfuse Docker Setup:  
  https://github.com/piyushgarg-dev/genai-cohort/blob/main/docker-compose.langfuse.yml

### Reference
- Contextual Retrieval (Anthropic):  
  https://www.anthropic.com/engineering/contextual-retrieval

---

## Lecture 8: LangGraph & Agent Routing

### Topics Covered
- LangGraph fundamentals
- State-based agent workflows
- Conditional routing between nodes
- Multi-step AI agents
- Deterministic agent execution

### Concepts Implemented
- StateGraph with typed state
- Router node for query classification
- Coding vs conversational agent routing
- JSON Schema–based structured outputs
- Groq LLM integration (without Instructor)
- LangSmith tracing for:
  - Graph nodes
  - Individual LLM calls

### Code Highlights
- Query classifier using Groq JSON Schema
- Coding assistant node
- Simple conversational assistant node
- Conditional routing via LangGraph
- Manual LLM tracing using `langsmith.trace`

### Key Takeaways
- LangGraph is ideal for **deterministic, production-grade agents**
- JSON Schema guarantees **safe, structured LLM outputs**
- Raw SDKs (Groq) require **manual LangSmith LLM tracing**
- This pattern avoids tool-calling pitfalls and scales cleanly

---

## 🛠️ Tech Stack & Tools
- Python
- LLMs (OpenAI / Open-Source Models)
- Groq API
- Hugging Face Transformers
- Ollama
- Vector Databases
- Graph Databases
- Docker & Docker Compose
- LangGraph
- LangSmith
- Langfuse

---

## 📖 How to Use This Repository
1. Follow lectures in sequence
2. Read notes before running code
3. Use Docker configs for local setup
4. Refer research papers for deeper understanding
5. Explore LangGraph examples for real-world agent orchestration

---

## ⭐ Acknowledgements
- Piyush Garg – GenAI Cohort
- Anthropic Research
- Open-source community
