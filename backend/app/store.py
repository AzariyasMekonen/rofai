import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


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
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_model: str):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embed_model = embed_model

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int, priority_boost: float = 0.05) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec.astype(np.float32).reshape(-1)
        # Hand-verified facts outrank noisy OCR text when similarity is close
        boost = np.array([priority_boost if c.priority else 0.0 for c in self.chunks], dtype=np.float32)
        ranked = np.argsort(-(scores + boost))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(index_dir / VECTORS_FILE, self.vectors)
        (index_dir / META_FILE).write_text(
            json.dumps({"embed_model": self.embed_model, "count": len(self.chunks)}), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore | None":
        chunks_path = index_dir / CHUNKS_FILE
        vectors_path = index_dir / VECTORS_FILE
        if not (chunks_path.exists() and vectors_path.exists()):
            return None
        with chunks_path.open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors = np.load(vectors_path)
        meta_path = index_dir / META_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(chunks, vectors, meta.get("embed_model", ""))
