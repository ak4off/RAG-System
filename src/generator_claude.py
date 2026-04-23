from pathlib import Path

import anthropic
import yaml

_client = anthropic.Anthropic()

_prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
with open(_prompts_path) as f:
    _prompts = yaml.safe_load(f)["rag_answer"]


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[Source {i+1}] {source} (page {page}):\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I cannot answer this question from the provided documents.",
            "sources": [],
        }

    context = _build_context(chunks)
    user_msg = _prompts["user_template"].format(context=context, query=query)

    response = _client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=_prompts["system"],
        messages=[{"role": "user", "content": user_msg}],
    )

    answer = response.content[0].text

    sources = [
        {
            "index": i + 1,
            "source": c["metadata"].get("source", "unknown"),
            "page": c["metadata"].get("page", "?"),
            "score": round(c.get("score", 0), 3),
            "excerpt": c["text"][:300].strip() + ("..." if len(c["text"]) > 300 else ""),
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": answer, "sources": sources}
