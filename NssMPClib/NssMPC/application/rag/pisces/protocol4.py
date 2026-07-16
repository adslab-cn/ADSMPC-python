"""Interactive Pisces Protocol 4: ∏MultLPSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from NssMPC.application.rag.pisces.psi import LabelCipher, kdf
from NssMPC.crypto.primitives.okvs import BinaryOKVS, OKVSTable
from NssMPC.crypto.primitives.oprf import (
    DHOPRFClient,
    DHOPRFParams,
    DHOPRFServer,
    OPRFBlindRequest,
    OPRFBlindResponse,
    OPRFBlindState,
)


@dataclass(frozen=True)
class Protocol4PublicSetup:
    table: OKVSTable
    num_docs: int
    prefix_size: int
    tf_size: int


class Protocol4Server:
    """Server side of Pisces ∏MultLPSI.

    The server builds the OKVS over encrypted term-frequency labels and answers
    OPRF blind-evaluation requests. It never receives the client's query tokens.
    """

    def __init__(
        self,
        *,
        okvs: BinaryOKVS | None = None,
        oprf_server: DHOPRFServer | None = None,
        cipher: LabelCipher | None = None,
        prefix_size: int = 16,
        tf_size: int = 8,
    ) -> None:
        self.okvs = okvs or BinaryOKVS()
        self.oprf_server = oprf_server or DHOPRFServer()
        self.cipher = cipher or LabelCipher()
        self.prefix_size = prefix_size
        self.tf_size = tf_size
        self.setup: Protocol4PublicSetup | None = None

    def build_setup(self, document_term_frequency: torch.Tensor) -> Protocol4PublicSetup:
        if document_term_frequency.dim() != 2:
            raise ValueError("document_term_frequency must have shape [vocab_size, num_docs]")

        vocab_size, num_docs = document_term_frequency.shape
        entries = []
        for token in range(vocab_size):
            row = document_term_frequency[token]
            nonzero_docs = torch.nonzero(row > 0, as_tuple=False).reshape(-1)
            for doc_tensor in nonzero_docs:
                doc_id = int(doc_tensor.item())
                tf = int(row[doc_id].item())
                entries.append((token, doc_id, tf))

        return self.build_setup_from_entries(num_docs=num_docs, entries=entries)

    def build_setup_from_entries(
        self,
        *,
        num_docs: int,
        entries: Iterable[tuple[int, int, int]],
    ) -> Protocol4PublicSetup:
        keys: list[bytes] = []
        values: list[bytes] = []
        prf_cache: dict[int, bytes] = {}
        for token, doc_id, tf in entries:
            if not 0 <= doc_id < num_docs:
                raise ValueError("doc_id is outside [0, num_docs)")
            if tf <= 0:
                continue
            prf_value = prf_cache.get(token)
            if prf_value is None:
                prf_value = self.oprf_server.evaluate_direct(token)
                prf_cache[token] = prf_value
            okvs_key = protocol4_okvs_key(doc_id, prf_value)
            label_key = protocol4_label_key(doc_id, prf_value)
            payload = (b"\x00" * self.prefix_size) + int(tf).to_bytes(self.tf_size, "big")
            keys.append(okvs_key)
            values.append(self.cipher.encrypt(label_key, payload))

        table = self.okvs.encode(keys, values)
        self.setup = Protocol4PublicSetup(table, num_docs, self.prefix_size, self.tf_size)
        return self.setup

    def evaluate_oprf(self, request: OPRFBlindRequest) -> OPRFBlindResponse:
        return self.oprf_server.evaluate(request)


class Protocol4Client:
    """Client side of Pisces ∏MultLPSI."""

    def __init__(
        self,
        *,
        okvs: BinaryOKVS | None = None,
        oprf_client: DHOPRFClient | None = None,
        cipher: LabelCipher | None = None,
    ) -> None:
        self.okvs = okvs or BinaryOKVS()
        self.oprf_client = oprf_client or DHOPRFClient()
        self.cipher = cipher or LabelCipher()
        self._query_tokens: torch.Tensor | None = None
        self._blind_state: OPRFBlindState | None = None

    def make_query(self, query_tokens: torch.Tensor) -> OPRFBlindRequest:
        if query_tokens.dim() != 1:
            query_tokens = query_tokens.reshape(-1)
        self._query_tokens = query_tokens
        items = [int(token.item()) for token in query_tokens]
        request, state = self.oprf_client.blind(items)
        self._blind_state = state
        return request

    def recover_term_frequencies(
        self,
        response: OPRFBlindResponse,
        setup: Protocol4PublicSetup,
    ) -> torch.Tensor:
        if self._query_tokens is None or self._blind_state is None:
            raise RuntimeError("make_query must be called before recover_term_frequencies")

        prf_values = self.oprf_client.finalize(self._blind_state, response)
        out = torch.zeros(
            setup.num_docs,
            self._query_tokens.numel(),
            dtype=torch.float32,
            device=self._query_tokens.device,
        )
        prefix = b"\x00" * setup.prefix_size
        for query_pos, prf_value in enumerate(prf_values):
            for doc_id in range(setup.num_docs):
                okvs_key = protocol4_okvs_key(doc_id, prf_value)
                label_key = protocol4_label_key(doc_id, prf_value)
                ciphertext = self.okvs.decode(setup.table, okvs_key)
                payload = self.cipher.decrypt(label_key, ciphertext)
                if payload[: setup.prefix_size] == prefix:
                    tf = int.from_bytes(payload[setup.prefix_size : setup.prefix_size + setup.tf_size], "big")
                    out[doc_id, query_pos] = float(tf)
        return out


def protocol4_okvs_key(doc_id: int, prf_value: bytes) -> bytes:
    return kdf(b"pisces-kdf0", doc_id.to_bytes(8, "big"), prf_value, size=32)


def protocol4_label_key(doc_id: int, prf_value: bytes) -> bytes:
    return kdf(b"pisces-kdf1", doc_id.to_bytes(8, "big"), prf_value, size=32)


def run_protocol4_server(party, protocol: Protocol4Server, document_term_frequency: torch.Tensor) -> Protocol4PublicSetup:
    """Run ∏MultLPSI on a NssMPClib server party.

    Message order:
    1. server -> client: public OKVS setup
    2. client -> server: blinded OPRF request
    3. server -> client: blinded OPRF response
    """

    setup = protocol.build_setup(document_term_frequency)
    party.send(setup)
    request = party.receive()
    response = protocol.evaluate_oprf(request)
    party.send(response)
    return setup


def run_protocol4_client(party, protocol: Protocol4Client, query_tokens: torch.Tensor) -> torch.Tensor:
    """Run ∏MultLPSI on a NssMPClib client party and return recovered TF."""

    setup = party.receive()
    request = protocol.make_query(query_tokens)
    party.send(request)
    response = party.receive()
    return protocol.recover_term_frequencies(response, setup)
