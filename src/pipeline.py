from src import vectorstore
from src.generator import generate
from src.ingest import ingest


class RAGPipeline:
    def __init__(self, persist_dir: str = "./chroma_db", k: int = 5):
        self.persist_dir = persist_dir
        self.k = k

    def index(self, paths: list[str]) -> None:
        chunks = ingest(paths)
        vectorstore.add(chunks, persist_dir=self.persist_dir)

    def query(self, question: str) -> dict:
        chunks = vectorstore.search(question, k=self.k, persist_dir=self.persist_dir)
        return generate(question, chunks)

    def clear(self) -> None:
        vectorstore.clear(persist_dir=self.persist_dir)
