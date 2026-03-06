# rag-langgraph-agent

# RAG LangGraph Research Agent (Chroma + Tools)

A production-style Retrieval Augmented Generation (RAG) agent built with **LangGraph**, **LangChain**, **ChromaDB**, and **tool routing**.

The agent follows a **KB-first retrieval policy** with optional **web search fallback**, providing structured answers through a FastAPI API and an interactive Streamlit UI.

This project demonstrates how to build **tool-using LLM agents with controllable reasoning flows**, a pattern commonly used in modern AI systems.

---

## Live Demo

Streamlit Interface
Interactive research assistant powered by a local LLM.

Demo UI:

![demo](docs/demo.png)

---

## Architecture

The system uses a **LangGraph state machine** to control the reasoning loop.

User Question
↓
KB Retrieval (ChromaDB)
↓
LLM Decision (LangGraph Agent)
↓
Tool Routing
• Knowledge Base
• Web Search (Tavily)
↓
Final Answer

Key design principles:

* **KB-first policy** to minimize hallucinations
* **Controlled tool usage**
* **Web search fallback only when retrieval is weak**
* **Budget-limited external tool calls**

---

## Project Structure

```
rag-langgraph-agent
│
├── backend
│   ├── app.py           # FastAPI API server
│   ├── agent.py         # LangGraph agent implementation
│   ├── config.py        # Environment configuration
│   └── requirements.txt
│
├── ui
│   ├── streamlit_app.py # Streamlit interface
│   └── requirements.txt
│
├── .env.example
└── README.md
```

---

## Key Features

### LangGraph Agent

* State-based reasoning loop
* Deterministic tool routing
* Recursion-limited execution

### Knowledge Base Retrieval

* ChromaDB vector store
* SentenceTransformers embeddings
* Distance-based relevance filtering

### Tool Integration

* `kb_query_tool`
* `search_web_tool`
* `kb_save_tool`

### Web Search

Tavily integration with usage budget control.

### Local LLM Support

Compatible with **vLLM OpenAI API server**.

---

## Technologies

* Python
* LangGraph
* LangChain
* ChromaDB
* SentenceTransformers
* FastAPI
* Streamlit
* vLLM
* Tavily Search API

---

## Setup

### 1 Install dependencies

Backend

```
pip install -r backend/requirements.txt
```

UI

```
pip install -r ui/requirements.txt
```

---

### 2 Environment Variables

Create an example environment file.

```
vim .env.example .env
```

Example `.env`

```
TAVILY_API_KEY=your_api_key
LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
LOCAL_OPENAI_MODEL=Qwen/Qwen2.5-3B-Instruct
LOCAL_OPENAI_API_KEY=EMPTY
TEMPERATURE=0.0
```

---

### 3 Start Local LLM (vLLM)

```
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

---

### 4 Run Backend API

```
python -m backend.app
```

API will start on

```
http://localhost:8080
```

API docs:

```
http://localhost:8080/docs
```

---

### 5 Run Streamlit UI

```
streamlit run ui/streamlit_app.py
```

UI will start on

```
http://localhost:8501
```

---

## Example Query

User question:

```
How can I reduce features in a dataset?
```

Agent workflow:

1. Query knowledge base
2. Evaluate retrieval quality
3. Optionally call web search
4. Produce final answer

---

## Why This Project Matters

This project demonstrates:

* Tool-using LLM agents
* LangGraph reasoning workflows
* Retrieval augmented generation
* Production-style AI architecture

These patterns are commonly used in:

* AI research assistants
* enterprise knowledge copilots
* autonomous agents
* search systems

---

## Future Improvements

* Hybrid retrieval (BM25 + vector)
* streaming responses
* tool call visualization
* evaluation benchmarks

---

## Author

Mohammadreza Kouchaki
AI / Machine Learning Engineer
