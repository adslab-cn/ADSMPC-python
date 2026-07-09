"""Pisces Protocol 1 private semantic retrieval coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from NssMPC import RingTensor
from NssMPC.application.rag.pisces.config import PiscesConfig
from NssMPC.application.rag.pisces.ops import semantic_inner_product_scores
from NssMPC.application.rag.pisces.pir import SudaPIRToShareResult, suda_pir_to_share
from NssMPC.application.rag.pisces.protocol3 import (
    Protocol3Client,
    Protocol3ClientMessage,
    Protocol3Document,
    Protocol3PublicSetup,
    Protocol3Server,
)
from NssMPC.application.rag.pisces.secure_sorting import SecureTopKAudit, secure_top_k_indicators


@dataclass(frozen=True)
class Protocol1CandidateResult:
    candidates: tuple[Protocol3Document, ...]
    candidate_indices: tuple[int, ...]
    candidate_mask: torch.Tensor
    setup: Protocol3PublicSetup


@dataclass(frozen=True)
class Protocol1SemanticResult:
    scores: Any
    indicators: Any
    documents: Any | None
    pir: SudaPIRToShareResult | None
    candidate_mask: torch.Tensor
    topk_audit: SecureTopKAudit | None = None


class Protocol1Server:
    """Server-side orchestration for Pisces Protocol 1."""

    def __init__(
        self,
        *,
        config: PiscesConfig | None = None,
        protocol3: Protocol3Server | None = None,
        empty_candidate_fallback: bool = True,
    ) -> None:
        self.config = config or PiscesConfig()
        self.protocol3 = protocol3 or Protocol3Server(
            threshold=self.config.hamming_threshold or 16,
            simhash_bits=self.config.simhash_bits,
        )
        self.empty_candidate_fallback = empty_candidate_fallback
        self.setup: Protocol3PublicSetup | None = None

    def build_filter_setup(
        self,
        document_embeddings: torch.Tensor,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Protocol3PublicSetup:
        self.setup = self.protocol3.build_setup_from_embeddings(document_embeddings, chunks=chunks)
        return self.setup

    def recover_candidates(self, message: Protocol3ClientMessage, *, num_docs: int) -> Protocol1CandidateResult:
        if self.setup is None:
            raise RuntimeError("build_filter_setup must be called before recover_candidates")
        candidates = self.protocol3.recover_candidates(message)
        candidate_indices = tuple(sorted(candidate.index for candidate in candidates))
        if not candidate_indices and self.empty_candidate_fallback:
            candidate_indices = tuple(range(num_docs))
        candidate_mask = torch.zeros(num_docs, dtype=torch.float32)
        if candidate_indices:
            candidate_mask[list(candidate_indices)] = 1.0
        return Protocol1CandidateResult(
            candidates=candidates,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            setup=self.setup,
        )


class Protocol1Client:
    """Client-side orchestration for Pisces Protocol 1."""

    def __init__(
        self,
        *,
        config: PiscesConfig | None = None,
        protocol3: Protocol3Client | None = None,
    ) -> None:
        self.config = config or PiscesConfig()
        self.protocol3 = protocol3 or Protocol3Client()

    def make_filter_query(self, query_embedding: torch.Tensor, setup: Protocol3PublicSetup) -> Protocol3ClientMessage:
        return self.protocol3.filter_from_embedding(query_embedding, setup)


def protocol1_finish_from_candidate_mask(
    query_embedding_share: Any,
    document_embedding_shares: Any,
    *,
    candidate_mask: torch.Tensor,
    top_k: int,
    document_payload_shares: Any | None = None,
    penalty: float = -1000000.0,
) -> Protocol1SemanticResult:
    """Run Protocol 1 fine scoring, top-k, and optional PIR-to-share.

    The candidate mask is the output of Protocol 3. Non-candidate scores receive
    a large negative public penalty before the exact top-k selection network.
    Plain integer payloads use the encrypted Suda OPR/OPE/OPI backend. Payloads
    that are already arithmetic secret shares keep the ASS fallback, because the
    Suda HE flow starts from a server-held plaintext database and a client
    encrypted query.
    """

    scores = semantic_inner_product_scores(query_embedding_share, document_embedding_shares).reshape(-1)
    if candidate_mask.numel() != scores.shape[-1]:
        raise ValueError("candidate_mask length must match number of document scores")
    if isinstance(scores, torch.Tensor):
        masked_scores = scores + (1.0 - candidate_mask.to(scores.device).float()) * penalty
    else:
        masked_scores = scores + RingTensor.convert_to_ring((1.0 - candidate_mask.float()) * penalty)
    topk = secure_top_k_indicators(masked_scores, top_k, return_audit=True)
    indicators = topk.indicators
    pir = suda_pir_to_share(indicators, document_payload_shares) if document_payload_shares is not None else None
    return Protocol1SemanticResult(
        scores=masked_scores,
        indicators=indicators,
        documents=pir.records if pir is not None else None,
        pir=pir,
        candidate_mask=candidate_mask,
        topk_audit=topk.audit,
    )


def run_protocol1_server_filter(
    party,
    protocol: Protocol1Server,
    document_embeddings: torch.Tensor,
    *,
    chunks: Sequence[Any] | None = None,
) -> Protocol1CandidateResult:
    setup = protocol.build_filter_setup(document_embeddings, chunks=chunks)
    party.send(setup)
    message = party.receive()
    result = protocol.recover_candidates(message, num_docs=document_embeddings.shape[0])
    party.send(result.candidate_mask)
    return result


def run_protocol1_client_filter(
    party,
    protocol: Protocol1Client,
    query_embedding: torch.Tensor,
) -> tuple[Protocol3ClientMessage, torch.Tensor]:
    setup = party.receive()
    message = protocol.make_filter_query(query_embedding, setup)
    party.send(message)
    candidate_mask = party.receive()
    return message, candidate_mask
