"""Check retrieval quality without the chat model.

Usage:
    uv run python -m app.query "When was the school founded?"
"""

import sys

from .config import get_settings
from .llm import embed
from .store import VectorStore


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    question = " ".join(sys.argv[1:])
    s = get_settings()
    store = VectorStore.load(s.index_dir)
    if store is None:
        print("No index found; run `python -m app.ingest` first.", file=sys.stderr)
        return 1
    for rank, hit in enumerate(store.search(embed([question])[0], s.top_k), start=1):
        c = hit.chunk
        where = f"{c.source} p.{c.page}" if c.page else c.source
        print(f"\n#{rank} score={hit.score:.3f} {where}{' [fact sheet]' if c.priority else ''}")
        print(c.text[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
