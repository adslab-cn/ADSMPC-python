"""Reusable retrieval operations for Pisces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

try:
    from NssMPC import ArithmeticSecretSharing, RingTensor
    from NssMPC.config import DEVICE
except ImportError:  # Allows plain unit tests outside the NssMPC runtime.
    ArithmeticSecretSharing = None
    RingTensor = None
    DEVICE = "cpu"

from NssMPC.application.rag.pisces.secure_sorting import secure_top_k_indicators


@dataclass(frozen=True)
class BM25Computation:
    scores: torch.Tensor
    df: torch.Tensor
    idf: torch.Tensor
    length_norm: torch.Tensor
    contributions: torch.Tensor


def simhash(vectors: torch.Tensor, projection: torch.Tensor | None = None, bits: int = 128) -> torch.Tensor:
    """Convert dense vectors to SimHash bits with a supplied or deterministic projection."""

    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)
    if projection is None:
        generator = torch.Generator(device=vectors.device)
        generator.manual_seed(20260607)
        projection = torch.randn(vectors.shape[-1], bits, generator=generator, device=vectors.device)
    return (vectors @ projection >= 0).to(torch.uint8)


def semantic_inner_product_scores(query_embedding: Any, document_embeddings: Any) -> Any:
    """Compute query-document dot products for ASS shares or plain tensors."""

    return (query_embedding * document_embeddings).sum(dim=-1)


def cosine_scores(query_embedding: Any, document_embeddings: Any, eps: float = 1e-8) -> Any:
    """Plain cosine scores; ASS callers should pre-normalize and use inner products."""

    if ArithmeticSecretSharing is not None and isinstance(query_embedding, ArithmeticSecretSharing):
        return semantic_inner_product_scores(query_embedding, document_embeddings)
    query_norm = torch.linalg.vector_norm(query_embedding, dim=-1, keepdim=True).clamp_min(eps)
    doc_norm = torch.linalg.vector_norm(document_embeddings, dim=-1, keepdim=True).clamp_min(eps)
    return semantic_inner_product_scores(query_embedding / query_norm, document_embeddings / doc_norm)


def bm25_scores_from_tf(
    query_term_frequencies: torch.Tensor,
    document_lengths: torch.Tensor,
    *,
    num_documents: int,
    average_document_length: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> torch.Tensor:
    """Compute BM25 from a [num_docs, query_terms] term-frequency matrix."""

    return bm25_components_from_tf(
        query_term_frequencies,
        document_lengths,
        num_documents=num_documents,
        average_document_length=average_document_length,
        k1=k1,
        b=b,
    ).scores


def bm25_components_from_tf(
    query_term_frequencies: torch.Tensor,
    document_lengths: torch.Tensor,
    *,
    num_documents: int,
    average_document_length: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> BM25Computation:
    """Compute BM25 and expose Protocol 2 intermediate values for audits."""

    tf = query_term_frequencies.float()
    df = (tf > 0).sum(dim=0).float()
    idf = torch.log1p((num_documents - df + 0.5) / (df + 0.5))
    length_factor = k1 * (1.0 - b + b * document_lengths.float().unsqueeze(-1) / average_document_length)
    contributions = idf.unsqueeze(0) * ((k1 + 1.0) * tf) / (tf + length_factor + 1e-8)
    return BM25Computation(
        scores=contributions.sum(dim=-1),
        df=df,
        idf=idf,
        length_norm=length_factor.squeeze(-1),
        contributions=contributions,
    )


def secure_bm25_matrix_scores(query_multihot_share: Any, bm25_matrix_share: Any) -> Any:
    """NssMPClib-friendly BM25-style linear scoring from test/rag.py."""

    return (query_multihot_share * bm25_matrix_share).sum(dim=0)


def secure_bm25_scores_from_shares(weighted_tf_share: Any, tf_share: Any, length_norm_share: Any) -> tuple[Any, Any]:
    """Compute Protocol 2 BM25 scores from secret-shared components.

    ``weighted_tf_share`` is a share of ``IDF(q_j) * (k1 + 1) * tf_ij`` with
    shape ``[num_docs, query_terms]``. ``tf_share`` has the same shape, and
    ``length_norm_share`` is a share of
    ``k1 * (1 - b + b * L_i / L_avg)`` with shape ``[num_docs]``.
    """

    denominator = tf_share + length_norm_share.unsqueeze(-1)
    contributions = weighted_tf_share / denominator
    return contributions.sum(dim=-1), contributions


def indicator_top_k(scores: Any, k: int) -> Any:
    """Return a secret-shared or plain [k, num_docs] Top-K indicator matrix."""

    return secure_top_k_indicators(scores, k)


def select_by_indicators(indicators: Any, values: Any) -> Any:
    """Select rows from ``values`` using [k, num_docs] indicators."""

    expanded = indicators
    while len(expanded.shape) < len(values.shape) + 1:
        expanded = expanded.unsqueeze(-1)
    return (expanded * values.unsqueeze(0)).sum(dim=1)


def default_average_length(document_lengths: torch.Tensor) -> float:
    return float(document_lengths.float().mean().item()) if document_lengths.numel() else math.nan
