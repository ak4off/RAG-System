from src import vectorstore
from src.generator import generate
from src.hybrid_retriever import HybridRetriever
from src.ingest import ingest
from src.reranker import rerank

REFUSAL = "I cannot answer this question from the provided documents."


class RAGPipelineV2:
    """
    Phase 2 pipeline:
      - Hybrid retrieval (BM25 + vector, RRF fusion)
      - Cross-encoder reranking
      - Citation integrity check (refuse if answer isn't grounded)
    """

    def __init__(self, persist_dir: str = "./chroma_db", k: int = 5, fetch_k: int = 20):
        self.persist_dir = persist_dir
        self.k = k
        self.fetch_k = fetch_k
        self._retriever = HybridRetriever(persist_dir=persist_dir, k=fetch_k)

    def index(self, paths: list[str]) -> None:
        chunks = ingest(paths)
        vectorstore.add(chunks, persist_dir=self.persist_dir)
        self._retriever._bm25 = None  # invalidate BM25 cache

    def query(self, question: str) -> dict:
        # Step 1: broad hybrid retrieval (fetch_k candidates)
        candidates = self._retriever.search(question)

        # Step 2: cross-encoder rerank, keep top k
        top_chunks = rerank(question, candidates, top_n=self.k)

        # Step 3: generate with citation enforcement
        result = generate(question, top_chunks)

        # Step 4: citation integrity - flag if model refused
        result["grounded"] = REFUSAL not in result["answer"]

        return result

    def clear(self) -> None:
        vectorstore.clear(persist_dir=self.persist_dir)
        self._retriever._bm25 = None
