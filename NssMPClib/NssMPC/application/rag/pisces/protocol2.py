"""Pisces Protocol 2: ∏PrivateBM25 private lexical retrieval.

Protocol 2 is the paper-level lexical retrieval protocol. Its first step
invokes Protocol 4 (∏MultLPSI) so the client obtains per-document query term
frequencies. Then the parties compute secret-shared BM25 scores and run secure
top-k. The payload retrieval boundary invokes the native Suda PIR-to-share
bridge after the lexical top-k ids are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import torch

from NssMPC.application.rag.pisces.config import PiscesConfig
from NssMPC.application.rag.pisces.ops import (
    default_average_length,
    secure_bm25_scores_from_shares,
)
from NssMPC.application.rag.pisces.pir import (
    SudaNativeClientState,
    SudaNativePIRClientShare,
    SudaNativePIRServerShare,
    SudaNativeServerState,
    suda_native_pir_to_share_client,
    suda_native_pir_to_share_server,
)
from NssMPC.application.rag.pisces.protocol4 import (
    Protocol4Client,
    Protocol4PublicSetup,
    Protocol4Server,
)
from NssMPC.application.rag.pisces.secure_sorting import SecureTopKAudit, secure_top_k_indicators
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


@dataclass(frozen=True)
class Protocol2WeightedTF:
    """Client-local Protocol 2 Line 2/3 plaintext intermediates."""

    term_frequencies: torch.Tensor
    df: torch.Tensor
    idf: torch.Tensor
    weighted_tf: torch.Tensor


@dataclass(frozen=True)
class Protocol2BM25Result:
    scores: Any
    contributions: Any


@dataclass(frozen=True)
class Protocol2LexicalResult:
    scores: Any
    indicators: Any
    topk_audit: SecureTopKAudit | None = None


class Protocol2Server:
    """Server-side orchestration for Pisces Protocol 2.

    Paper mapping:
    - Line 1 invokes Protocol 4 through ``build_lpsi_setup`` and
      ``evaluate_lpsi``.
    - Line 3 prepares the server-side length-normalization term.
    - Line 4 computes secret-shared BM25 scores.
    - Line 5 secure top-k is exposed through ``finish_from_scores``.
    """

    def __init__(
        self,
        *,
        config: PiscesConfig | None = None,
        protocol4: Protocol4Server | None = None,
    ) -> None:
        self.config = config or PiscesConfig()
        self.protocol4 = protocol4 or Protocol4Server()
        self.setup: Protocol4PublicSetup | None = None

    @classmethod
    def paper_defaults(
        cls,
        *,
        config: PiscesConfig | None = None,
        okvs_expansion: float = 2.4,
        okvs_seed: bytes = b"rag-protocol4-okvs",
        oprf_secret_key: int | None = None,
    ) -> "Protocol2Server":
        """Construct Protocol 2 with the paper-level Protocol 4 backend."""

        oprf_params = DHOPRFParams()
        oprf_server = (
            DHOPRFServer(params=oprf_params, secret_key=oprf_secret_key)
            if oprf_secret_key is not None
            else DHOPRFServer(params=oprf_params)
        )
        return cls(
            config=config,
            protocol4=Protocol4Server(
                okvs=BinaryOKVS(expansion=okvs_expansion, seed=okvs_seed),
                oprf_server=oprf_server,
            ),
        )

    def build_lpsi_setup(self, document_term_frequency: torch.Tensor) -> Protocol4PublicSetup:
        self.setup = self.protocol4.build_setup(document_term_frequency)
        return self.setup

    def build_lpsi_setup_from_entries(
        self,
        *,
        num_docs: int,
        entries: Iterable[tuple[int, int, int]],
    ) -> Protocol4PublicSetup:
        self.setup = self.protocol4.build_setup_from_entries(num_docs=num_docs, entries=entries)
        return self.setup

    def evaluate_lpsi(self, request):
        return self.protocol4.evaluate_oprf(request)

    def length_norm_plain(self, document_lengths: torch.Tensor) -> torch.Tensor:
        return self.config.bm25_k1 * (
            1.0
            - self.config.bm25_b
            + self.config.bm25_b * document_lengths.float() / default_average_length(document_lengths)
        )

    def score_from_shares(self, weighted_tf_share: Any, tf_share: Any, length_norm_share: Any) -> Protocol2BM25Result:
        scores, contributions = secure_bm25_scores_from_shares(
            weighted_tf_share,
            tf_share,
            length_norm_share,
        )
        return Protocol2BM25Result(scores=scores, contributions=contributions)

    def finish_from_scores(
        self,
        scores: Any,
        *,
        top_k: int | None = None,
    ) -> Protocol2LexicalResult:
        topk = secure_top_k_indicators(scores, top_k or self.config.top_k, return_audit=True)
        return Protocol2LexicalResult(
            scores=scores,
            indicators=topk.indicators,
            topk_audit=topk.audit,
        )

    def retrieve_topk_documents(
        self,
        party: Any,
        document_payload: torch.Tensor,
        *,
        top_k: int | None = None,
        server_state: SudaNativeServerState | None = None,
    ) -> SudaNativePIRServerShare:
        """Run Protocol 2 Line 6 batch PIR-to-share on the server side."""

        return suda_native_pir_to_share_server(
            party,
            document_payload,
            selected_count=top_k or self.config.top_k,
            server_state=server_state,
        )


class Protocol2Client:
    """Client-side orchestration for Pisces Protocol 2."""

    def __init__(
        self,
        *,
        config: PiscesConfig | None = None,
        protocol4: Protocol4Client | None = None,
    ) -> None:
        self.config = config or PiscesConfig()
        self.protocol4 = protocol4 or Protocol4Client()

    @classmethod
    def paper_defaults(
        cls,
        *,
        config: PiscesConfig | None = None,
    ) -> "Protocol2Client":
        """Construct Protocol 2 client with the paper-level Protocol 4 backend."""

        return cls(config=config, protocol4=Protocol4Client(oprf_client=DHOPRFClient(params=DHOPRFParams())))

    def make_lpsi_query(self, query_tokens: torch.Tensor):
        return self.protocol4.make_query(query_tokens)

    def recover_term_frequencies(self, response, setup: Protocol4PublicSetup) -> torch.Tensor:
        return self.protocol4.recover_term_frequencies(response, setup)

    def weighted_tf_plain(
        self,
        term_frequencies: torch.Tensor,
        *,
        num_docs: int,
    ) -> Protocol2WeightedTF:
        df = (term_frequencies > 0).sum(dim=0).float()
        idf = torch.log1p((num_docs - df + 0.5) / (df + 0.5))
        weighted_tf = idf.unsqueeze(0) * (self.config.bm25_k1 + 1.0) * term_frequencies
        return Protocol2WeightedTF(
            term_frequencies=term_frequencies,
            df=df,
            idf=idf,
            weighted_tf=weighted_tf,
        )

    def score_from_shares(self, weighted_tf_share: Any, tf_share: Any, length_norm_share: Any) -> Protocol2BM25Result:
        scores, contributions = secure_bm25_scores_from_shares(
            weighted_tf_share,
            tf_share,
            length_norm_share,
        )
        return Protocol2BM25Result(scores=scores, contributions=contributions)

    def finish_from_scores(
        self,
        scores: Any,
        *,
        top_k: int | None = None,
    ) -> Protocol2LexicalResult:
        topk = secure_top_k_indicators(scores, top_k or self.config.top_k, return_audit=True)
        return Protocol2LexicalResult(
            scores=scores,
            indicators=topk.indicators,
            topk_audit=topk.audit,
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
        """Run Protocol 2 Line 6 batch PIR-to-share on the client side."""

        return suda_native_pir_to_share_client(
            party,
            topk_ids,
            dtype=dtype,
            device=device,
            previous_state=previous_state,
        )
