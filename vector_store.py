from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config
from embeddings import get_embedding_model
def _to_documents(chunks: list[dict]) -> list[Document]:
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "heading": chunk.get("heading"),
                    "section": chunk.get("section"),
                    "token_count": chunk.get("token_count"),
                    "language": chunk.get("language", "unknown"),
                    "keywords": chunk.get("keywords", []),
                },
            )
        )
    return documents
def build_index(chunks: list[dict]) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")
    documents = _to_documents(chunks)
    return FAISS.from_documents(
        documents,
        get_embedding_model(),
    )
def save_index(index: FAISS, path: str = config.FAISS_INDEX_PATH) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    index.save_local(path)
def load_index(path: str = config.FAISS_INDEX_PATH) -> FAISS:
    if not index_exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return FAISS.load_local(
        path,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
def index_exists(path: str = config.FAISS_INDEX_PATH) -> bool:
    path = Path(path)
    return (
        (path / "index.faiss").exists()
        and
        (path / "index.pkl").exists()
    )
