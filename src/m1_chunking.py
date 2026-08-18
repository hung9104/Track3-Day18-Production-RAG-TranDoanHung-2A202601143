from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


_SEMANTIC_MODEL = None


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  WARNING: Skipping {os.path.basename(fp)}: scanned PDF has no text layer (OCR required).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    # Implementation outline: semantic chunking
    # 1. from sentence_transformers import SentenceTransformer
    #    from numpy import dot
    #    from numpy.linalg import norm
    # 2. metadata = metadata or {}
    # 3. Split text thành sentences: re.split(r'(?<=[.!?])\s+|\n\n', text)
    # 4. model = SentenceTransformer("all-MiniLM-L6-v2")
    #    embeddings = model.encode(sentences)
    # 5. cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-9)
    # 6. Duyệt từ sentence[1]:
    #      - sim(embedding[i-1], embedding[i]) < threshold → tách chunk mới
    #      - else: gộp vào chunk hiện tại
    # 7. Return [Chunk(text=joined_group, metadata={..., "strategy": "semantic"})]
    global _SEMANTIC_MODEL
    metadata = metadata or {}
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if part.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic"})]

    try:
        from numpy import dot
        from numpy.linalg import norm
        from sentence_transformers import SentenceTransformer

        if _SEMANTIC_MODEL is None:
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _SEMANTIC_MODEL.encode(sentences, show_progress_bar=False)

        groups = [[sentences[0]]]
        for index in range(1, len(sentences)):
            similarity = float(
                dot(embeddings[index - 1], embeddings[index])
                / (norm(embeddings[index - 1]) * norm(embeddings[index]) + 1e-9)
            )
            if similarity < threshold:
                groups.append([])
            groups[-1].append(sentences[index])
    except Exception as exc:
        print(f"  WARNING: Semantic model unavailable, using paragraph fallback: {exc}")
        groups = [[part] for part in sentences]

    return [
        Chunk(text="\n\n".join(group), metadata={**metadata, "strategy": "semantic"})
        for group in groups if group
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    # Implementation outline: hierarchical chunking
    # 1. metadata = metadata or {}
    # 2. Split text bằng "\n\n" → paragraphs
    # 3. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars):
    #      pid = f"parent_{len(parents)}"
    #      parents.append(Chunk(text=..., metadata={..., "chunk_type": "parent", "parent_id": pid}))
    # 4. Mỗi parent → split thành children (mỗi child ≤ child_size chars):
    #      children.append(Chunk(text=..., metadata={..., "chunk_type": "child"}, parent_id=pid))
    # 5. return (parents, children)
    metadata = metadata or {}

    def split_to_size(value: str, size: int) -> list[str]:
        if size <= 0:
            raise ValueError("Chunk size must be greater than zero")
        pieces = []
        remaining = value.strip()
        while len(remaining) > size:
            boundary = remaining.rfind(" ", 0, size + 1)
            if boundary <= 0:
                boundary = size
            pieces.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    parent_texts = []
    current = ""
    for paragraph in paragraphs:
        for piece in split_to_size(paragraph, parent_size):
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > parent_size:
                parent_texts.append(current)
                current = piece
            else:
                current = candidate
    if current:
        parent_texts.append(current)

    parents, children = [], []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parent_meta = {**metadata, "chunk_type": "parent", "parent_id": parent_id}
        parents.append(Chunk(text=parent_text, metadata=parent_meta))
        for child_index, child_text in enumerate(split_to_size(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "child_index": child_index},
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    # Implementation outline: structure-aware chunking
    # 1. metadata = metadata or {}
    # 2. sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    # 3. Duyệt sections:
    #      - Nếu match header (^#{1,3}\s+): lưu header hiện tại, tạo chunk cho content trước đó
    #      - Else: gộp vào content hiện tại
    # 4. Return [Chunk(text=header+content, metadata={..., "section": header, "strategy": "structure"})]
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        stripped = text.strip()
        return ([Chunk(text=stripped, metadata={**metadata, "section": "", "strategy": "structure"})]
                if stripped else [])

    chunks = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(text=preamble, metadata={**metadata, "section": "", "strategy": "structure"}))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start():end].strip()
        chunks.append(Chunk(
            text=section_text,
            metadata={**metadata, "section": match.group(2).strip(), "strategy": "structure"},
        ))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
