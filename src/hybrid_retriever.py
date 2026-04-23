from rank_bm25 import BM25Okapi

from src import vectorstore


class HybridRetriever:
    """
    Combines BM25 keyword search with dense vector search.
    Scores are fused using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, persist_dir: str = "./chroma_db", k: int = 5, rrf_k: int = 60):
        self.persist_dir = persist_dir
        self.k = k
        self.rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] | None = None

    def _load_corpus(self) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = chromadb.PersistentClient(path=self.persist_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        col = client.get_or_create_collection(vectorstore.COLLECTION, embedding_function=ef)
        data = col.get()

        self._corpus = [
            {"text": t, "metadata": m}
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        tokenized = [doc["text"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank + 1)

    def search(self, query: str) -> list[dict]:
        if self._bm25 is None:
            self._load_corpus()

        # BM25 results
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # Vector results
        vec_results = vectorstore.search(query, k=self.k * 2, persist_dir=self.persist_dir)

        # Build lookup: text -> vector rank
        vec_rank = {r["text"]: rank for rank, r in enumerate(vec_results)}

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked[: self.k * 2]):
            rrf[idx] = rrf.get(idx, 0.0) + self._rrf_score(rank)

        for rank, r in enumerate(vec_results):
            # Find corpus index matching this text
            for idx, doc in enumerate(self._corpus):
                if doc["text"] == r["text"]:
                    rrf[idx] = rrf.get(idx, 0.0) + self._rrf_score(rank)
                    break

        top_indices = sorted(rrf, key=rrf.__getitem__, reverse=True)[: self.k]

        return [
            {
                "text": self._corpus[i]["text"],
                "metadata": self._corpus[i]["metadata"],
                "score": rrf[i],
            }
            for i in top_indices
        ]
