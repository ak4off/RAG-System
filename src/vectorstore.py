import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from langchain.schema import Document
from langchain_core.documents import Document

COLLECTION = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _get_collection(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)


def add(chunks: list[Document], persist_dir: str = "./chroma_db") -> None:
    col = _get_collection(persist_dir)
    existing = set(col.get()["ids"])

    ids, texts, metas = [], [], []
    for c in chunks:
        uid = f"{c.metadata.get('source', 'doc')}_{c.metadata.get('chunk_id', 0)}"
        if uid in existing:
            continue
        ids.append(uid)
        texts.append(c.page_content)
        metas.append({k: str(v) for k, v in c.metadata.items()})

    if ids:
        col.add(documents=texts, metadatas=metas, ids=ids)
        print(f"[vectorstore] added {len(ids)} chunks (skipped {len(chunks) - len(ids)} duplicates)")
    else:
        print("[vectorstore] all chunks already indexed")


def search(query: str, k: int = 5, persist_dir: str = "./chroma_db") -> list[dict]:
    col = _get_collection(persist_dir)
    results = col.query(query_texts=[query], n_results=k)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })
    return chunks


def clear(persist_dir: str = "./chroma_db") -> None:
    client = chromadb.PersistentClient(path=persist_dir)
    client.delete_collection(COLLECTION)
    print("[vectorstore] cleared")
