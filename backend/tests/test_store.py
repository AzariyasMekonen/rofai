from pathlib import Path

import numpy as np

from app.store import Chunk, VectorStore


def _store() -> VectorStore:
    chunks = [
        Chunk("a", "library history", "a.pdf", 1),
        Chunk("b", "sports day", "b.pdf", 2),
        Chunk("c", "founding facts", "facts.md", None, [1986], priority=True),
    ]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.96, 0.28, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return VectorStore(chunks, vectors, "test-model")


def test_search_returns_most_similar_first():
    hits = _store().search(np.array([0, 1, 0], dtype=np.float32), k=2, priority_boost=0.0)
    assert [h.chunk.id for h in hits] == ["b", "c"]


def test_priority_boost_breaks_near_ties():
    query = np.array([1, 0, 0], dtype=np.float32)
    assert _store().search(query, k=1, priority_boost=0.0)[0].chunk.id == "a"
    assert _store().search(query, k=1, priority_boost=0.05)[0].chunk.id == "c"


def test_save_and_load_roundtrip(tmp_path: Path):
    store = _store()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded is not None
    assert loaded.embed_model == "test-model"
    assert [c.id for c in loaded.chunks] == ["a", "b", "c"]
    assert loaded.chunks[2].years == [1986]
    np.testing.assert_allclose(loaded.vectors, store.vectors)


def test_load_missing_index_returns_none(tmp_path: Path):
    assert VectorStore.load(tmp_path) is None
