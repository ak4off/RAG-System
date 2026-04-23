from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-scores chunks by running query+chunk through a cross-encoder.
    More accurate than bi-encoder similarity but slower — run after
    an initial broad retrieval.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [
        {**chunk, "rerank_score": float(score)}
        for score, chunk in ranked[:top_n]
    ]
