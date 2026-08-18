from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, re, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _model_cache = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Implementation outline: lazy-load and share the cross-encoder.
            # from sentence_transformers import CrossEncoder
            # self._model = CrossEncoder(self.model_name)
            #
            # ⚠️ LƯU Ý: Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            if self.model_name in self._model_cache:
                self._model = self._model_cache[self.model_name]
            else:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name)
                    self._model_cache[self.model_name] = self._model
                except Exception as exc:
                    print(f"  WARNING: Cross-encoder unavailable, using lexical fallback: {exc}")
                    self._model_cache[self.model_name] = None
        return self._model

    @staticmethod
    def _lexical_scores(query: str, documents: list[dict]) -> list[float]:
        tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        scores = []
        for document in documents:
            document_tokens = set(re.findall(r"\w+", document.get("text", "").lower(), flags=re.UNICODE))
            overlap = len(tokens & document_tokens)
            scores.append(overlap / max(len(tokens), 1))
        return scores

    @classmethod
    def _fallback_scores(cls, query: str, documents: list[dict]) -> list[float]:
        """Deterministic fallback used only when the model cannot be loaded."""
        lexical = cls._lexical_scores(query, documents)
        return [score + float(document.get("score", 0.0)) * 1e-3
                for score, document in zip(lexical, documents)]

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # Implementation outline: cross-encoder scoring and descending ranking
        # 1. if not documents: return []
        # 2. model = self._load_model()
        # 3. pairs = [(query, doc["text"]) for doc in documents]
        # 4. scores = model.predict(pairs)
        # 5. if isinstance(scores, (int, float)): scores = [scores]
        # 6. scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        # 7. Return [RerankResult(text=..., original_score=doc.get("score", 0.0),
        #            rerank_score=float(score), metadata=..., rank=i)
        #            for i, (score, doc) in enumerate(scored[:top_k])]
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if model is None:
            scores = self._fallback_scores(query, documents)
        else:
            pairs = [(query, document.get("text", "")) for document in documents]
            try:
                predicted = model.predict(pairs, show_progress_bar=False)
                if isinstance(predicted, (int, float)):
                    scores = [float(predicted)]
                else:
                    scores = [float(score) for score in predicted]
                # Some reranker checkpoints return saturated sigmoid scores. Normalize
                # within the candidate set and use a small lexical tie-breaker.
                low, high = min(scores), max(scores)
                if high > low:
                    scores = [(score - low) / (high - low) for score in scores]
                lexical = self._lexical_scores(query, documents)
                scores = [score + 0.1 * lexical_score
                          for score, lexical_score in zip(scores, lexical)]
            except Exception as exc:
                print(f"  WARNING: Cross-encoder prediction failed, using lexical fallback: {exc}")
                scores = self._fallback_scores(query, documents)

        scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional implementation: from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
