from dataclasses import dataclass


@dataclass(frozen=True)
class PiscesConfig:
    """Shape and protocol knobs for a Pisces retrieval run."""

    top_k: int = 1
    semantic_candidates: int | None = None
    simhash_bits: int = 64
    hamming_threshold: int | None = None
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
