"""Pisces Protocol 1: ∏PrivateSS private semantic retrieval.

Protocol 1 is the paper-level semantic retrieval protocol. Its coarse matching
step invokes Protocol 3 (∏Oblivious Filter), then the fine stage computes
secret-shared semantic scores and secure top-k indicators. The payload retrieval
boundary invokes the native Suda PIR-to-share bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from NssMPC import RingTensor
from NssMPC.application.rag.pisces.config import PiscesConfig
from NssMPC.application.rag.pisces.ops import semantic_inner_product_scores
from NssMPC.application.rag.pisces.pir import (
    SudaNativeClientState,
    SudaNativePIRClientShare,
    SudaNativePIRServerShare,
    SudaNativeServerState,
    suda_native_pir_to_share_client,
    suda_native_pir_to_share_server,
)
from NssMPC.application.rag.pisces.protocol3 import (
    AdditivePaillier,
    Protocol3Client,
    Protocol3ClientMessage,
    Protocol3Document,
    Protocol3PublicSetup,
    Protocol3Server,
)
from NssMPC.application.rag.pisces.secure_sorting import SecureTopKAudit, secure_top_k_indicators
from NssMPC.crypto.primitives.okvs import BinaryOKVS


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
    candidate_mask: torch.Tensor
    topk_audit: SecureTopKAudit | None = None


class Protocol1Server:
    """Server-side orchestration for Pisces Protocol 1.

    Paper mapping:
    - Line 1/2 setup and coarse matching are delegated to Protocol 3.
    - Line 3/4 fine scoring and secure top-k are handled by
      ``finish_from_candidate_mask``.
    - Line 5 PIR-to-share is exposed through ``retrieve_topk_documents``.
    """

    def __init__(
        self,
        *,
        config: PiscesConfig | None = None,
        protocol3: Protocol3Server | None = None,
    ) -> None:
        self.config = config or PiscesConfig()
        self.protocol3 = protocol3 or Protocol3Server(
            threshold=self.config.hamming_threshold or 16,
            simhash_bits=self.config.simhash_bits,
        )
        self.setup: Protocol3PublicSetup | None = None

    @classmethod
    def paper_defaults(
        cls,
        *,
        config: PiscesConfig | None = None,
        okvs_expansion: float = 3.0,
        okvs_seed: bytes = b"rag-protocol3-okvs",
        paillier_key_size: int = 256,
        threshold: int | None = None,
        projection_count: int = 160,
        seed: bytes = b"rag-protocol3",
        simhash_bits: int | None = None,
    ) -> "Protocol1Server":
        """Construct Protocol 1 with the paper-level Protocol 3 backend."""

        cfg = config or PiscesConfig()
        return cls(
            config=cfg,
            protocol3=Protocol3Server(
                okvs=BinaryOKVS(expansion=okvs_expansion, seed=okvs_seed),
                he=AdditivePaillier(key_size=paillier_key_size),
                threshold=threshold if threshold is not None else (cfg.hamming_threshold or 16),
                projection_count=projection_count,
                seed=seed,
                simhash_bits=simhash_bits if simhash_bits is not None else cfg.simhash_bits,
            ),
        )

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
        candidate_mask = torch.zeros(num_docs, dtype=torch.float32)
        if candidate_indices:
            candidate_mask[list(candidate_indices)] = 1.0
        return Protocol1CandidateResult(
            candidates=candidates,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            setup=self.setup,
        )

    def finish_from_candidate_mask(
        self,
        query_embedding_share: Any,
        document_embedding_shares: Any,
        *,
        candidate_mask: torch.Tensor,
        top_k: int | None = None,
        penalty: float = -1000000.0,
    ) -> Protocol1SemanticResult:
        """Run Protocol 1 Line 3/4 on server shares."""

        return protocol1_finish_from_candidate_mask(
            query_embedding_share,
            document_embedding_shares,
            candidate_mask=candidate_mask,
            top_k=top_k or self.config.top_k,
            penalty=penalty,
        )

    def retrieve_topk_documents(
        self,
        party: Any,
        document_payload: torch.Tensor,
        *,
        top_k: int | None = None,
        server_state: SudaNativeServerState | None = None,
    ) -> SudaNativePIRServerShare:
        """Run Protocol 1 Line 5 batch PIR-to-share on the server side."""

        return suda_native_pir_to_share_server(
            party,
            document_payload,
            selected_count=top_k or self.config.top_k,
            server_state=server_state,
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

    @classmethod
    def paper_defaults(
        cls,
        *,
        config: PiscesConfig | None = None,
        shuffle_seed: bytes = b"rag-protocol3-client",
    ) -> "Protocol1Client":
        """Construct Protocol 1 client with the paper-level Protocol 3 backend."""

        return cls(config=config, protocol3=Protocol3Client(shuffle_seed=shuffle_seed))

    def make_filter_query(self, query_embedding: torch.Tensor, setup: Protocol3PublicSetup) -> Protocol3ClientMessage:
        return self.protocol3.filter_from_embedding(query_embedding, setup)

    def finish_from_candidate_mask(
        self,
        query_embedding_share: Any,
        document_embedding_shares: Any,
        *,
        candidate_mask: torch.Tensor,
        top_k: int | None = None,
        penalty: float = -1000000.0,
    ) -> Protocol1SemanticResult:
        """Run Protocol 1 Line 3/4 on client shares."""

        return protocol1_finish_from_candidate_mask(
            query_embedding_share,
            document_embedding_shares,
            candidate_mask=candidate_mask,
            top_k=top_k or self.config.top_k,
            penalty=penalty,
        )

    def retrieve_topk_documents(
        self,
        party: Any,
        topk_ids: Any,
        *,
        dtype: torch.dtype = torch.float32,
        device: Any = "cpu",
        previous_state: SudaNativeClientState | None = None,
    ) -> SudaNativePIRClientShare:
        """Run Protocol 1 Line 5 batch PIR-to-share on the client side."""

        return suda_native_pir_to_share_client(
            party,
            topk_ids,
            dtype=dtype,
            device=device,
            previous_state=previous_state,
        )


def protocol1_finish_from_candidate_mask(
    query_embedding_share: Any,
    document_embedding_shares: Any,
    *,
    candidate_mask: torch.Tensor,
    top_k: int,
    penalty: float = -1000000.0,
) -> Protocol1SemanticResult:
    """Run Protocol 1 fine scoring and secure top-k.

    The candidate mask is the output of Protocol 3. Non-candidate scores receive
    a large negative public penalty before the exact top-k selection network.
    Payload retrieval is intentionally separated into ``retrieve_topk_documents``
    so the paper-level native Suda PIR-to-share bridge is the only online
    payload path.
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
    return Protocol1SemanticResult(
        scores=masked_scores,
        indicators=indicators,
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
