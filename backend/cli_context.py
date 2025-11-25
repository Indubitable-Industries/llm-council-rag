# Simple CLI to inspect what context would be sent for a query.
# Usage examples:
#   python -m backend.cli_context --conversation <id> --query "How do we start the server?"
#   python -m backend.cli_context --conversation <id> --query "..." --manual-file backend/main.py

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.rag_lmstudio import get_context  # noqa: E402


def load_manual_files(paths):
    items = []
    for p in paths:
        path_obj = Path(p)
        try:
            content = path_obj.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
            continue
        items.append({
            "path": str(path_obj),
            "content": content,
            "source_type": "manual_cli",
        })
    return items


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Inspect context for a conversation/query")
    parser.add_argument("--conversation", required=True, help="Conversation id")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--manual-file", action="append", default=[], help="File paths to force include (can repeat)")

    args = parser.parse_args()

    manual_items = load_manual_files(args.manual_file)
    context_text, entries = get_context(args.conversation, args.query, manual_items)

    print("=== Query ===")
    print(args.query)
    print()

    if manual_items:
        print(f"Manual context supplied: {len(manual_items)} file(s). RAG skipped.")
    else:
        print("RAG retrieval used.")
    print()

    print("=== Context summary ===")
    for entry in entries:
        tag = entry.get("source_type", "").upper() or "RAG"
        score = entry.get("score")
        meta = [f"lines={entry.get('lines')}", f"bytes={entry.get('bytes')}", f"tokens~{entry.get('est_tokens')}"]
        if score is not None:
            meta.append(f"score={score:.3f}")
        print(f"[{tag}] {entry.get('source')} ({', '.join(meta)})")

    print()
    print("=== Context text sent ===")
    print(context_text)


if __name__ == "__main__":
    main()
