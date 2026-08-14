from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

from ingestion import load_document
from nlp import process_pages
from chunking import chunk_pages
from embeddings import embed_chunks
from vector_store import build_index, save_index, load_index, index_exists
from retrieval import hybrid_search
from reranker import rerank
from llm import generate_answer


class RAGState(TypedDict):
    file_path: str
    query: str
    pages: List[dict]
    chunks: List[dict]
    index: Optional[object]
    retrieved: List[dict]
    reranked: List[dict]
    answer: str


def ingest_node(state):
    pages = load_document(state["file_path"])
    return {"pages": pages}


def nlp_node(state):
    pages = process_pages(state["pages"])
    return {"pages": pages}


def chunk_node(state):
    chunks = chunk_pages(state["pages"])
    return {"chunks": chunks}


def embed_and_store_node(state):
    chunks = embed_chunks(state["chunks"])
    index = build_index(chunks)
    save_index(index)
    return {"chunks": chunks, "index": index}


def retrieve_node(state):
    index = state["index"] or load_index()
    retrieved = hybrid_search(state["query"], state["chunks"], index)
    return {"retrieved": retrieved}


def rerank_node(state):
    reranked = rerank(state["query"], state["retrieved"])
    return {"reranked": reranked}


def generate_node(state):
    answer = generate_answer(state["query"], state["reranked"])
    return {"answer": answer}


def build_ingestion_graph():
    graph = StateGraph(RAGState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("nlp", nlp_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed_and_store", embed_and_store_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "nlp")
    graph.add_edge("nlp", "chunk")
    graph.add_edge("chunk", "embed_and_store")
    graph.add_edge("embed_and_store", END)

    return graph.compile()


def build_query_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def ingest_and_index(file_path):
    app = build_ingestion_graph()
    result = app.invoke({"file_path": file_path})
    return result["chunks"], result["index"]


def answer_query(query, chunks, index=None):
    app = build_query_graph()
    result = app.invoke({"query": query, "chunks": chunks, "index": index})
    return result["answer"]
