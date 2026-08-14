from sentence_transformers import CrossEncoder

import config

_reranker_model = CrossEncoder(config.RERANKER_MODEL, device=config.EMBEDDING_DEVICE)


def get_reranker():
    return _reranker_model


def rerank(query, chunks):
    if not config.USE_RERANKER or not chunks:
        return chunks[:config.RERANK_TOP_N]

    model = get_reranker()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:config.RERANK_TOP_N]]
