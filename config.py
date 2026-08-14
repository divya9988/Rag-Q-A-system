# Rag-Q-A-system"""
config.py
---------
Central configuration for the Smart RAG Question Answering System.
This file contains only configurable settings.
No business logic should be written here.
Every module should import values from this file instead of
hardcoding constants.
"""

# ============================================================================
# Application
# ============================================================================
APP_TITLE = "Smart RAG Q/A System"

# ============================================================================
# LLM
# ============================================================================
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# ============================================================================
# Embeddings
# ============================================================================
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"          # "cuda" if GPU is available
EMBEDDING_BATCH_SIZE = 16

# ============================================================================
# Vector Database
# ============================================================================
VECTOR_DB = "faiss"
FAISS_INDEX_PATH = "storage/faiss_index"

# ============================================================================
# Chunking
# ============================================================================
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# ============================================================================
# Retrieval
# ============================================================================
TOP_K = 8
RELATIVE_SCORE_MARGIN = 0.4
MMR_DIVERSITY = 0.3

# Hybrid Search
USE_HYBRID_SEARCH = True
FAISS_TOP_K = 15
BM25_TOP_K = 15
HYBRID_SEMANTIC_WEIGHT = 0.60
HYBRID_BM25_WEIGHT = 0.40

# ============================================================================
# Reranker
# ============================================================================
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANK_TOP_N = 5

# ============================================================================
# OCR
# ============================================================================
OCR_ENGINE = "paddleocr"
OCR_FALLBACK_ENGINE = "tesseract"
OCR_LANGUAGE = "en"
TESSERACT_LANGUAGE = "eng"
TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PREFIX = r"C:\Program Files\Tesseract-OCR\tessdata"

# ============================================================================
# Document Support
# ============================================================================
SUPPORTED_FILE_TYPES = [
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "txt",
]
MAX_DOCUMENT_PAGES = 100

# ============================================================================
# NLP
# ============================================================================
UNICODE_NORMALIZATION_FORM = "NFKC"
DETECT_LANGUAGE = True

# ============================================================================
# Storage
# ============================================================================
DATA_DIRECTORY = "data"
DOCUMENT_DIRECTORY = "documents"
VECTOR_DIRECTORY = "storage"

# ============================================================================
# Miscellaneous
# ============================================================================
RANDOM_SEED = 42
