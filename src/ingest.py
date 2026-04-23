from pathlib import Path

import tiktoken
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 700       # tokens
CHUNK_OVERLAP = 100    # tokens

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_file(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".pdf":
        loader = PyPDFLoader(str(p))
    elif p.suffix in (".txt", ".md"):
        loader = TextLoader(str(p), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", p.name)
    return docs


def chunk(docs: list[Document]) -> list[Document]:
    splitter = _get_splitter()
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def ingest(paths: list[str]) -> list[Document]:
    all_chunks = []
    for path in paths:
        docs = load_file(path)
        all_chunks.extend(chunk(docs))
    print(f"[ingest] {len(all_chunks)} chunks from {len(paths)} file(s)")
    return all_chunks
