import argparse
import json

from src.pipeline import RAGPipeline


def fmt_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("ANSWER\n")
    print(result["answer"])
    print("\n" + "-" * 60)
    print("SOURCES\n")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['source']} | page {s['page']} | score {s['score']}")
        print(f"      {s['excerpt'][:120]}...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("--db", default="./chroma_db", help="ChromaDB persist path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index files into the vector store")
    p_index.add_argument("files", nargs="+")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--json", action="store_true", help="Output raw JSON")

    sub.add_parser("clear", help="Wipe the vector store")

    args = parser.parse_args()
    pipeline = RAGPipeline(persist_dir=args.db, k=getattr(args, "k", 5))

    if args.cmd == "index":
        pipeline.index(args.files)

    elif args.cmd == "query":
        result = pipeline.query(args.question)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            fmt_result(result)

    elif args.cmd == "clear":
        pipeline.clear()


if __name__ == "__main__":
    main()
