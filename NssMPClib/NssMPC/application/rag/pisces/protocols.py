"""Protocol interfaces and local fallbacks for Pisces.

The concrete Pisces paper protocols are:
- Protocol 3: oblivious filter over SimHash/Hamming distance.
- Protocol 4: multi-instance labeled PSI for BM25 term frequencies.
- batch PIR-to-share for retrieving selected chunks.

The classes below provide deterministic local substitutes so the retrieval
pipeline can be exercised before those cryptographic protocols are fully ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ObliviousFilter(Protocol):
    def candidates(
        self,
        query_bits: torch.Tensor,
        document_bits: torch.Tensor,
        *,
        threshold: int | None,
        limit: int | None,
    ) -> torch.Tensor:
        """Return candidate document indices for the semantic fine stage."""


class MultiInstanceLabeledPSI(Protocol):
    def term_frequencies(
        self,
        query_tokens: torch.Tensor,
        document_term_frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-document term frequencies for every query token."""


@dataclass
class LocalHammingFilter:
    """Plain local filter with the same input/output contract as Protocol 3."""

    def candidates(
        self,
        query_bits: torch.Tensor,
        document_bits: torch.Tensor,
        *,
        threshold: int | None,
        limit: int | None,
    ) -> torch.Tensor:
        if query_bits.dim() != 1:
            query_bits = query_bits.reshape(-1)
        if document_bits.dim() != 2:
            raise ValueError("document_bits must have shape [num_docs, simhash_bits]")

        distances = (document_bits.to(torch.bool) != query_bits.to(torch.bool)).sum(dim=-1)
        if threshold is not None:
            candidate_idx = torch.nonzero(distances <= threshold, as_tuple=False).reshape(-1)
        else:
            candidate_idx = torch.arange(document_bits.shape[0], device=document_bits.device)

        if limit is not None and candidate_idx.numel() > limit:
            candidate_distances = distances[candidate_idx]
            order = torch.argsort(candidate_distances, stable=True)[:limit]
            candidate_idx = candidate_idx[order]

        if candidate_idx.numel() == 0:
            fallback_count = min(limit or 1, document_bits.shape[0])
            candidate_idx = torch.argsort(distances, stable=True)[:fallback_count]

        return candidate_idx


@dataclass
class LocalTermFrequencyPSI:
    """Plain local replacement for Protocol 4.

    ``document_term_frequency`` is expected to be a dense matrix with shape
    [vocab_size, num_docs], matching the BM25 matrix shape used in test/rag.py.
    """

    def term_frequencies(
        self,
        query_tokens: torch.Tensor,
        document_term_frequency: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.dim() != 1:
            query_tokens = query_tokens.reshape(-1)
        return document_term_frequency[query_tokens.long()].transpose(0, 1)

