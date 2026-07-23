"""Pisces Protocol 4 support: OKVS-backed ∏MultLPSI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import hmac
import os
from typing import Iterable, Protocol

import torch

from NssMPC.crypto.primitives.okvs import BinaryOKVS, OKVSTable


class OPRFClient(Protocol):
    """Client-side OPRF interface.

    A production implementation should obtain PRF outputs through a real OPRF
    protocol. The local implementation below is only for same-process tests.
    """

    def evaluate(self, item: int | str | bytes) -> bytes:
        ...


@dataclass
class LocalHMACOPRF:
    """Same-process OPRF stand-in used for correctness tests."""

    key: bytes

    def evaluate(self, item: int | str | bytes) -> bytes:
        return hmac.new(self.key, encode_item(item), hashlib.sha256).digest()


@dataclass
class AESCTRLabelCipher:
    """AES-CTR label encryption for Pisces ∏MultLPSI.

    Pisces describes AES encryption for ``0^lambda || tf`` labels. We use a
    deterministic AES-CTR nonce derived from the per-label key so OKVS values
    stay fixed-length and decode remains stateless.
    """

    def encrypt(self, key: bytes, plaintext: bytes) -> bytes:
        Cipher, algorithms, modes = _aes_ctr_classes()

        aes_key = key[:32]
        nonce = kdf(b"pisces-aes-ctr-nonce", key, size=16)
        encryptor = Cipher(algorithms.AES(aes_key), modes.CTR(nonce)).encryptor()
        return encryptor.update(plaintext) + encryptor.finalize()

    def decrypt(self, key: bytes, ciphertext: bytes) -> bytes:
        return self.encrypt(key, ciphertext)


LabelCipher = AESCTRLabelCipher


@lru_cache(maxsize=1)
def _aes_ctr_classes():
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:
        raise RuntimeError("cryptography is required for AES label encryption") from exc
    return Cipher, algorithms, modes


def encode_item(item: int | str | bytes) -> bytes:
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return int(item).to_bytes(16, "big", signed=True)


def kdf(domain: bytes, *parts: bytes, size: int = 32) -> bytes:
    material = domain + b"".join(len(part).to_bytes(4, "big") + part for part in parts)
    blocks = []
    counter = 0
    while sum(len(block) for block in blocks) < size:
        blocks.append(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    return b"".join(blocks)[:size]


class OKVSMultiInstanceLabeledPSI:
    """Pisces ∏MultLPSI data flow over a binary OKVS.

    This implements the server OKVS setup, client decode/decrypt path, and
    prefix validation. OPRF is represented by ``OPRFClient`` and defaults to a
    local HMAC-based stand-in for tests.
    """

    def __init__(
        self,
        *,
        okvs: BinaryOKVS | None = None,
        oprf_key: bytes | None = None,
        oprf_client: OPRFClient | None = None,
        cipher: LabelCipher | None = None,
        prefix_size: int = 16,
        tf_size: int = 8,
    ) -> None:
        self.okvs = okvs or BinaryOKVS()
        self.oprf_key = oprf_key or os.urandom(32)
        self.oprf_client = oprf_client or LocalHMACOPRF(self.oprf_key)
        self.cipher = cipher or LabelCipher()
        self.prefix_size = prefix_size
        self.tf_size = tf_size
        self.table: OKVSTable | None = None
        self.num_docs = 0

    @property
    def value_size(self) -> int:
        return self.prefix_size + self.tf_size

    def setup(self, document_term_frequency: torch.Tensor) -> OKVSTable:
        if document_term_frequency.dim() != 2:
            raise ValueError("document_term_frequency must have shape [vocab_size, num_docs]")

        vocab_size, num_docs = document_term_frequency.shape
        keys: list[bytes] = []
        values: list[bytes] = []

        for token in range(vocab_size):
            row = document_term_frequency[token]
            nonzero_docs = torch.nonzero(row > 0, as_tuple=False).reshape(-1)
            for doc_tensor in nonzero_docs:
                doc_id = int(doc_tensor.item())
                tf = int(row[doc_id].item())
                prf_value = self._server_prf(token)
                okvs_key = self._okvs_key(doc_id, prf_value)
                label_key = self._label_key(doc_id, prf_value)
                payload = (b"\x00" * self.prefix_size) + tf.to_bytes(self.tf_size, "big")
                keys.append(okvs_key)
                values.append(self.cipher.encrypt(label_key, payload))

        self.table = self.okvs.encode(keys, values)
        self.num_docs = num_docs
        return self.table

    def term_frequencies(
        self,
        query_tokens: torch.Tensor,
        document_term_frequency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.table is None:
            if document_term_frequency is None:
                raise ValueError("document_term_frequency is required before setup")
            self.setup(document_term_frequency)

        if query_tokens.dim() != 1:
            query_tokens = query_tokens.reshape(-1)

        out = torch.zeros(self.num_docs, query_tokens.numel(), dtype=torch.float32, device=query_tokens.device)
        for query_pos, token_tensor in enumerate(query_tokens):
            token = int(token_tensor.item())
            prf_value = self.oprf_client.evaluate(token)
            for doc_id in range(self.num_docs):
                okvs_key = self._okvs_key(doc_id, prf_value)
                label_key = self._label_key(doc_id, prf_value)
                ciphertext = self.okvs.decode(self.table, okvs_key)
                payload = self.cipher.decrypt(label_key, ciphertext)
                if payload[: self.prefix_size] == b"\x00" * self.prefix_size:
                    tf = int.from_bytes(payload[self.prefix_size : self.prefix_size + self.tf_size], "big")
                    out[doc_id, query_pos] = float(tf)
        return out

    def _server_prf(self, token: int) -> bytes:
        return hmac.new(self.oprf_key, encode_item(token), hashlib.sha256).digest()

    def _okvs_key(self, doc_id: int, prf_value: bytes) -> bytes:
        return kdf(b"pisces-kdf0", doc_id.to_bytes(8, "big"), prf_value, size=32)

    def _label_key(self, doc_id: int, prf_value: bytes) -> bytes:
        return kdf(b"pisces-kdf1", doc_id.to_bytes(8, "big"), prf_value, size=32)
