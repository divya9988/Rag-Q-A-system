import numpy as np
from rank_bm25 import BM25Okapi

import config


def _build_bm25(chunks):
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def _bm25_search(bm25, chunks, query, top_k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _semantic_search(index, query, top_k):
    results = index.similarity_search_with_score(query, k=top_k)
    return [
        (
            {
                "text": doc.page_content,
                "chunk_id": doc.metadata["chunk_id"],
                "document_name": doc.metadata["document_name"],
                "page_number": doc.metadata["page_number"],
                "heading": doc.metadata.get("heading"),
                "section": doc.metadata.get("section"),
                "language": doc.metadata.get("language", "unknown"),
            },
            -distance,
        )
        for doc, distance in results
    ]


def _normalize_scores(results):
    if not results:
        return {}
    scores = [score for _, score in results]
    min_score, max_score = min(scores), max(scores)
    span = max_score - min_score or 1e-9
    return {
        chunk["chunk_id"]: (score - min_score) / span
        for chunk, score in results
    }


def _cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b) or 1e-9
    return float(np.dot(a, b) / denom)


def _mmr_select(candidates, top_k, diversity=0.3):
    if not candidates:
        return []

    selected = [candidates[0]]
    remaining = candidates[1:]

    while remaining and len(selected) < top_k:
        def mmr_score(item):
            chunk, relevance = item
            if not chunk.get("embedding"):
                return relevance
            max_similarity = max(
                _cosine_similarity(chunk["embedding"], s[0]["embedding"])
                for s in selected
                if s[0].get("embedding")
            ) if any(s[0].get("embedding") for s in selected) else 0.0
            return (1 - diversity) * relevance - diversity * max_similarity

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return [chunk for chunk, _ in selected]


def hybrid_search(query, chunks, index):
    semantic_results = _semantic_search(index, query, config.FAISS_TOP_K)
    bm25 = _build_bm25(chunks)
    bm25_results = _bm25_search(bm25, chunks, query, config.BM25_TOP_K)

    semantic_scores = _normalize_scores(semantic_results)
    bm25_scores = _normalize_scores(bm25_results)

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    all_ids = set(semantic_scores) | set(bm25_scores)

    combined = []
    for chunk_id in all_ids:
        semantic_score = semantic_scores.get(chunk_id, 0.0)
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        final_score = (
            config.HYBRID_SEMANTIC_WEIGHT * semantic_score
            + config.HYBRID_BM25_WEIGHT * bm25_score
        )
        combined.append((chunk_lookup[chunk_id], final_score))

    combined.sort(key=lambda x: x[1], reverse=True)

    if not combined:
        return []

    top_score = combined[0][1]
    candidate_pool = [
        item for item in combined
        if item[1] >= top_score * config.RELATIVE_SCORE_MARGIN
    ]
    if len(candidate_pool) < config.TOP_K:
        candidate_pool = combined[:max(config.TOP_K, len(candidate_pool))]

    return _mmr_select(candidate_pool, config.TOP_K, diversity=config.MMR_DIVERSITY)
