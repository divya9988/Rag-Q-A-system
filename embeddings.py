from langchain_huggingface import HuggingFaceEmbeddings
import config
# Load the embedding model only once
_embedding_model = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={
        "device": config.EMBEDDING_DEVICE,
    },
    encode_kwargs={
        "batch_size": config.EMBEDDING_BATCH_SIZE,
        "normalize_embeddings": True,
    },
)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model.
    """
    return _embedding_model
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for document chunks.
    Args:
        chunks: List of chunk dictionaries.
    Returns:
        Same list with an 'embedding' field added.
    """
    if not chunks:
        return []
    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.embed_documents(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks
