import os
import json
import uuid
from typing import Annotated, TypedDict, List, Any, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.config import (
    LOCAL_OPENAI_BASE_URL,
    LOCAL_OPENAI_MODEL,
    LOCAL_OPENAI_API_KEY,
    TEMPERATURE,
    PERSIST_DIR,
    COLLECTION,
    EMBED_MODEL,
    TAVILY_API_KEY,
    TAVILY_MAX_SEARCH_CALLS_PER_RUN,
    TAVILY_MAX_RESULTS_PER_CALL,
    KB_WEAK_DISTANCE_THRESHOLD,
)

# ------------------------------
# LLM
# ------------------------------
llm = ChatOpenAI(
    base_url=LOCAL_OPENAI_BASE_URL,
    api_key=LOCAL_OPENAI_API_KEY,
    model=LOCAL_OPENAI_MODEL,
    temperature=TEMPERATURE,
)

SYSTEM_PROMPT = """
You are a tool-using RAG agent.

Policy:
- The system will ALWAYS provide KB hits first (kb_query_tool output).
- If KB hits are weak, call search_web_tool and extract exact facts.
- Otherwise, answer using KB hits.
- Keep answers concise and factual.
"""

# ------------------------------
# ChromaDB + Embedder
# ------------------------------
chroma = chromadb.PersistentClient(path=PERSIST_DIR)
embedder = SentenceTransformer(EMBED_MODEL)
kb = chroma.get_or_create_collection(COLLECTION)

def kb_save(items: list) -> str:
    """Save web results into Chroma KB."""
    ids, docs, metas = [], [], []
    for i, item in enumerate(items):
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        content = (item.get("content") or "")[:800]

        text = f"{title}\n{url}\n{content}".strip()
        if not text:
            continue

        ids.append(f"web_{abs(hash(url))}_{i}")
        docs.append(text)
        metas.append({"source": url if url else "web"})

    if not docs:
        return "saved 0"

    emb = embedder.encode(docs, normalize_embeddings=True).tolist()
    kb.upsert(ids=ids, embeddings=emb, documents=docs, metadatas=metas)
    return f"saved {len(docs)}"

@tool
def kb_save_tool(items: list) -> str:
    """Save web results into Chroma KB."""
    return kb_save(items)

@tool
def kb_query_tool(query: str, top_k: int = 3) -> list:
    """Query Chroma KB and return hits with distance (lower is better)."""
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    results = kb.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for d, m, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": d,
                "source": (m or {}).get("source", "kb"),
                "distance": dist,
            }
        )
    return hits

# ------------------------------
# Tavily
# ------------------------------
tavily = TavilyClient(TAVILY_API_KEY) if TAVILY_API_KEY else None

@tool
def search_web_tool(query: str, k: int = 1) -> list:
    """Search Tavily and return results. Also saves them into KB."""
    if tavily is None:
        return [{"title": "", "url": None, "content": "TAVILY_API_KEY_MISSING"}]

    try:
        r = tavily.search(query=query, max_results=k, search_depth="advanced")
    except Exception as e:
        return [{"title": "", "url": None, "content": str(e)}]

    items = []
    for item in r.get("results", []):
        items.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
            }
        )

    kb_save(items)
    return items

# ---------------------------
# LangGraph State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    tavily_calls: int
    max_steps: int
    blocked: bool

tools = [kb_query_tool, search_web_tool, kb_save_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# ----------------------------
# Helpers
# ----------------------------
def _extract_last_user_question(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def _kb_hits_are_weak(messages: List[Any]) -> bool:
    """
    Reads the latest KB hits (ToolMessage JSON) and decides if weak.
    We consider the best (min) distance; if above threshold => weak.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                hits = json.loads(msg.content)
                if isinstance(hits, list) and hits and "distance" in hits[0]:
                    best = min(h.get("distance", 1.0) for h in hits)
                    return best > KB_WEAK_DISTANCE_THRESHOLD
            except Exception:
                continue
    return True

# ----------------------------
# Node 0: KB-first enforced (NO LLM needed)
# ----------------------------
# def kb_first_node(state: AgentState) -> Dict[str, Any]:
#     question = _extract_last_user_question(state["messages"])
#     if not question:
#         return {}

#     hits = kb_query_tool.invoke({"query": question, "top_k": 3})

#     # Put KB results into the messages as a tool message so the LLM sees them
#     return {
#         "messages": [
#             ToolMessage(
#                 tool_call_id=f"kb_{uuid.uuid4().hex[:8]}",
#                 content=json.dumps(hits),
#             )
#         ]
#     }

def kb_first_node(state: AgentState) -> Dict[str, Any]:
    question = _extract_last_user_question(state["messages"])
    if not question:
        return {}

    hits = kb_query_tool.invoke({"query": question, "top_k": 3})
    return {
        "messages": [
            SystemMessage(content="KB hits (JSON):\n" + json.dumps(hits))
        ]
    }

# ----------------------------
# Node 1: Agent (LLM)
# ----------------------------
# def agent_node(state: AgentState) -> Dict[str, Any]:
#     msgs = state["messages"]

#     # Provide a short hint based on KB quality (helps the model behave)
#     weak = _kb_hits_are_weak(msgs)
#     hint = (
#         "KB hits look WEAK. You should call search_web_tool."
#         if weak
#         else "KB hits look STRONG. Answer using KB; do NOT call web search."
#     )

#     prompt_msgs = [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + hint)] + msgs
#     ai = llm_with_tools.invoke(prompt_msgs)

#     # --- QWEN XML tool_call fix (your original patch, kept) ---
#     if "<tool_call>" in (ai.content or "") and not getattr(ai, "tool_calls", None):
#         try:
#             content = ai.content
#             start = content.find("<tool_call>") + len("<tool_call>")
#             end = content.find("</tool_call>")
#             json_str = content[start:end].strip()
#             tool_data = json.loads(json_str)

#             ai.tool_calls = [
#                 {
#                     "name": tool_data["name"],
#                     "args": tool_data["arguments"],
#                     "id": f"call_{uuid.uuid4().hex[:8]}",
#                 }
#             ]
#             ai.content = ""
#         except Exception as e:
#             print(f"XML Parsing failed: {e}")
#     # ----------------------------------------------------------

#     return {"messages": [ai]}

def agent_node(state: AgentState) -> Dict[str, Any]:
    msgs = state["messages"]

    tavily_allowed = state["tavily_calls"] < TAVILY_MAX_SEARCH_CALLS_PER_RUN
    tools_allowed = [kb_query_tool, kb_save_tool] + ([search_web_tool] if tavily_allowed else [])

    llm_local = llm.bind_tools(tools_allowed)

    hint = (
        "KB hits look WEAK. You should call search_web_tool."
        if _kb_hits_are_weak(msgs) and tavily_allowed
        else "Answer using KB. Web search is unavailable or not needed."
    )

    prompt_msgs = [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + hint)] + msgs
    ai = llm_local.invoke(prompt_msgs)

    # keep your Qwen XML patch here if needed...
    return {"messages": [ai]}

# ----------------------------
# Node 2: Tavily budget guard
# ----------------------------
def maybe_block_tavily(state: AgentState) -> Dict[str, Any]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    updates: Dict[str, Any] = {"blocked": False}

    if not tool_calls:
        return updates

    for tc in tool_calls:
        if tc.get("name") == "search_web_tool":
            if state["tavily_calls"] >= TAVILY_MAX_SEARCH_CALLS_PER_RUN:
                updates["blocked"] = True
                updates["messages"] = [
                    ToolMessage(
                        tool_call_id=tc["id"],
                        content='[{"title":"","url":null,"content":"TAVILY_BUDGET_EXCEEDED"}]',
                    )
                ]
                return updates

            # allow and charge budget
            updates["tavily_calls"] = state["tavily_calls"] + 1
            return updates

    return updates

# def route_after_guard(state: AgentState):
#     if state.get("blocked"):
#         return "agent"  # allow LLM to continue with injected tool output
#     last = state["messages"][-1]
#     if getattr(last, "tool_calls", None):
#         return "tools"
#     return END

def route_after_guard(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

# ----------------------------
# Build graph
# ----------------------------
graph = StateGraph(AgentState)

graph.add_node("kb_first", kb_first_node)
graph.add_node("agent", agent_node)
graph.add_node("guard", maybe_block_tavily)
graph.add_node("tools", tool_node)

graph.set_entry_point("kb_first")
graph.add_edge("kb_first", "agent")
graph.add_edge("agent", "guard")
graph.add_conditional_edges("guard", route_after_guard, {"tools": "tools", "agent": "agent", END: END})
graph.add_edge("tools", "agent")

app_graph = graph.compile()

# ----------------------------
# Runner
# ----------------------------
def run(question: str, max_steps: int = 6) -> str:
    state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "tavily_calls": 0,
        "max_steps": max_steps,
        "blocked": False,
    }

    out = app_graph.invoke(state, config={"recursion_limit": max_steps * 2})

    for msg in reversed(out["messages"]):
        if isinstance(msg, AIMessage):
            text = (msg.content or "").strip()
            if text:
                return text

    return "No Final answer."
