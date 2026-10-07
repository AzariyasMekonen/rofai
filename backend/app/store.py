import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it its me my of on or "
    "our tell that the their there this to was were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


class KeywordIndex:
    """Okapi BM25 over chunk texts; used when no embedding service is configured."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = set(tokenize(query))
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in terms:
                tf = doc.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            out[i] = s
        return out


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    years: list[int] = field(default_factory=list)
    priority: bool = False


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    """Archive chunks with optional embedding vectors; keyword search always works."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None, embed_model: str):
        if vectors is not None and len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32) if vectors is not None else None
        self.embed_model = embed_model if vectors is not None else ""
        self._keywords: KeywordIndex | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None

    def _priority_mask(self) -> np.ndarray:
        return np.array([c.priority for c in self.chunks], dtype=bool)

    def search(self, query_vec: np.ndarray, k: int, priority_boost: float = 0.05) -> list[Hit]:
        if not self.chunks or self.vectors is None:
            return []
        scores = self.vectors @ query_vec.astype(np.float32).reshape(-1)
        # Hand-verified facts outrank noisy OCR text when similarity is close
        ranked = np.argsort(-(scores + self._priority_mask() * priority_boost))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked]

    def search_text(self, query: str, k: int, priority_factor: float = 1.15) -> list[Hit]:
        if not self.chunks:
            return []
        if self._keywords is None:
            self._keywords = KeywordIndex([c.text for c in self.chunks])
        scores = self._keywords.scores(query)
        boosted = np.where(self._priority_mask(), scores * priority_factor, scores)
        ranked = np.argsort(-boosted)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        vectors_path = index_dir / VECTORS_FILE
        if self.vectors is not None:
            np.save(vectors_path, self.vectors)
        else:
            # A stale vectors file would no longer line up with the new chunks
            vectors_path.unlink(missing_ok=True)
        (index_dir / META_FILE).write_text(
            json.dumps({"embed_model": self.embed_model, "count": len(self.chunks)}), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore | None":
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.exists():
            return None
        with chunks_path.open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors_path = index_dir / VECTORS_FILE
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        meta_path = index_dir / META_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(chunks, vectors, meta.get("embed_model", ""))
