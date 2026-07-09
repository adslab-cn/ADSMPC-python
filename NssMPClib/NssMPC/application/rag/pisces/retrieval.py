"""Composable Pisces retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PiscesConfig
from .ops import (
    bm25_scores_from_tf,
    cosine_scores,
    default_average_length,
    indicator_top_k,
    secure_bm25_matrix_scores,
    semantic_inner_product_scores,
    simhash,
)
from .pir import suda_pir_to_share
from .protocols import LocalHammingFilter, LocalTermFrequencyPSI, MultiInstanceLabeledPSI, ObliviousFilter


@dataclass
class RetrievalResult:
    semantic_scores: object
    lexical_scores: object | None
    semantic_indicators: object
    lexical_indicators: object | None
    semantic_documents: object | None = None
    lexical_documents: object | None = None


class PiscesRetriever:
    """Dual-path retrieval coordinator for executable Pisces demos.

    This class wires Protocol 1/2 scoring and top-k boundaries. Any returned
    payload documents are selected by the executable Suda PIR-to-share boundary.
    Plain integer payloads use encrypted OPR/OPE/OPI; ASS payloads keep the ASS
    fallback.
    """

    def __init__(
        self,
        config: PiscesConfig | None = None,
        *,
        oblivious_filter: ObliviousFilter | None = None,
        labeled_psi: MultiInstanceLabeledPSI | None = None,
    ) -> None:
        self.config = config or PiscesConfig()
        self.oblivious_filter = oblivious_filter or LocalHammingFilter()
        self.labeled_psi = labeled_psi or LocalTermFrequencyPSI()

    def semantic_path(
        self,
        query_embedding,
        document_embeddings,
        *,
        document_payload=None,
        projection: torch.Tensor | None = None,
    ) -> tuple[object, object, object | None]:
        """Run coarse-to-fine semantic retrieval.

        Plain tensors use SimHash coarse filtering. Secret-shared tensors go
        directly to the fine stage because the full oblivious filter is pending.
        """

        candidate_idx = None
        candidate_embeddings = document_embeddings
        if isinstance(query_embedding, torch.Tensor) and isinstance(document_embeddings, torch.Tensor):
            query_bits = simhash(query_embedding, projection, self.config.simhash_bits).reshape(-1)
            document_bits = simhash(document_embeddings, projection, self.config.simhash_bits)
            candidate_idx = self.oblivious_filter.candidates(
                query_bits,
                document_bits,
                threshold=self.config.hamming_threshold,
                limit=self.config.semantic_candidates,
            )
            candidate_embeddings = document_embeddings[candidate_idx]
            scores = cosine_scores(query_embedding, candidate_embeddings)
        else:
            scores = semantic_inner_product_scores(query_embedding, candidate_embeddings)

        indicators = indicator_top_k(scores, self.config.top_k)
        docs = None
        if document_payload is not None:
            payload = document_payload[candidate_idx] if candidate_idx is not None else document_payload
            docs = suda_pir_to_share(indicators, payload).records
        return scores, indicators, docs

    def lexical_path_from_matrix(self, query_multihot_share, bm25_matrix_share, *, document_payload=None):
        """Run the NssMPClib-friendly lexical path from a dense BM25 matrix."""

        scores = secure_bm25_matrix_scores(query_multihot_share, bm25_matrix_share)
        return self.lexical_path_from_scores(scores, document_payload=document_payload)

    def lexical_path_from_scores(self, scores, *, document_payload=None):
        """Run lexical Top-K selection from precomputed score shares."""

        indicators = indicator_top_k(scores, self.config.top_k)
        docs = suda_pir_to_share(indicators, document_payload).records if document_payload is not None else None
        return scores, indicators, docs

    def lexical_path_plain(
        self,
        query_tokens: torch.Tensor,
        document_term_frequency: torch.Tensor,
        document_lengths: torch.Tensor,
        *,
        document_payload=None,
    ):
        """Plain BM25 path with the same boundary as Pisces Protocol 2."""

        tf = self.labeled_psi.term_frequencies(query_tokens, document_term_frequency)
        scores = bm25_scores_from_tf(
            tf,
            document_lengths,
            num_documents=document_term_frequency.shape[1],
            average_document_length=default_average_length(document_lengths),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        indicators = indicator_top_k(scores, self.config.top_k)
        docs = suda_pir_to_share(indicators, document_payload).records if document_payload is not None else None
        return scores, indicators, docs

    def retrieve_from_secure_matrices(
        self,
        query_embedding_share,
        document_embedding_shares,
        query_multihot_share,
        bm25_matrix_share,
        *,
        document_token_shares=None,
    ) -> RetrievalResult:
        semantic_scores, semantic_indicators, semantic_docs = self.semantic_path(
            query_embedding_share,
            document_embedding_shares,
            document_payload=document_token_shares,
        )
        lexical_scores, lexical_indicators, lexical_docs = self.lexical_path_from_matrix(
            query_multihot_share,
            bm25_matrix_share,
            document_payload=document_token_shares,
        )
        return RetrievalResult(
            semantic_scores=semantic_scores,
            lexical_scores=lexical_scores,
            semantic_indicators=semantic_indicators,
            lexical_indicators=lexical_indicators,
            semantic_documents=semantic_docs,
            lexical_documents=lexical_docs,
        )
