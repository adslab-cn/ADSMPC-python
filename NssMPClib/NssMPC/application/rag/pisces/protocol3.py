"""Pisces Protocol 3: ∏Oblivious Filter.

This follows the protocol structure from the Pisces paper: projected SimHash
exact matching, 2-out-of-T Shamir reconstruction in the encrypted domain,
and server-side candidate recovery after decrypting shuffled ciphertexts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Sequence

import torch

from NssMPC.application.rag.pisces.ops import simhash
from NssMPC.crypto.primitives.homomorphic_encryption.paillier import Paillier
from NssMPC.crypto.primitives.okvs import BinaryOKVS, OKVSTable

try:  # Optional acceleration for Paillier big-integer arithmetic.
    import gmpy2 as _gmpy2
except ImportError:  # pragma: no cover - exercised on environments without gmpy2
    _gmpy2 = None


@dataclass(frozen=True)
class Protocol3Document:
    index: int
    simhash_bits: tuple[int, ...]
    chunk: Any


@dataclass(frozen=True)
class Protocol3PublicSetup:
    public_key: tuple[int, int]
    masks: tuple[tuple[int, ...], ...]
    projection_points: tuple[int, ...]
    table: OKVSTable
    simhash_bits: int
    threshold: int
    projection_weight: int
    ciphertext_size: int
    projection_collision_count: int = 0


@dataclass(frozen=True)
class Protocol3ClientState:
    decoded_projection_ciphertexts: tuple[int, ...]

    @property
    def decoded_ciphertexts(self) -> tuple[int, ...]:
        return self.decoded_projection_ciphertexts


@dataclass(frozen=True)
class Protocol3ClientMessage:
    shuffled_secret_ciphertexts: tuple[int, ...]


class AdditivePaillier:
    """Small randomized Paillier wrapper with additive HE operations."""

    def __init__(self, *, key_size: int = 256, rng: random.Random | None = None) -> None:
        self.paillier = Paillier()
        self.paillier.gen_keys(key_size)
        self.rng = rng or random.SystemRandom()

    @property
    def public_key(self) -> tuple[int, int]:
        if self.paillier.public_key is None:
            raise RuntimeError("Paillier keypair has not been generated")
        return self.paillier.public_key

    @property
    def private_key(self) -> tuple[int, int]:
        if self.paillier._private_key is None:
            raise RuntimeError("Paillier keypair has not been generated")
        return self.paillier._private_key

    @property
    def modulus(self) -> int:
        return self.public_key[0]

    @property
    def modulus_squared(self) -> int:
        n = self.modulus
        return n * n

    def encrypt(self, plaintext: int) -> int:
        return paillier_encrypt(plaintext, self.public_key, self.rng)

    def decrypt(self, ciphertext: int) -> int:
        return self.paillier.decrypt(ciphertext % self.modulus_squared)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["rng"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = random.SystemRandom()


class Protocol3Server:
    """Server side of Pisces ∏Oblivious Filter."""

    def __init__(
        self,
        *,
        okvs: BinaryOKVS | None = None,
        he: AdditivePaillier | None = None,
        threshold: int = 16,
        projection_count: int = 160,
        seed: bytes = b"pisces-protocol3",
        projection: torch.Tensor | None = None,
        simhash_bits: int = 128,
    ) -> None:
        self.okvs = okvs or BinaryOKVS(expansion=2.4)
        self.he = he or AdditivePaillier()
        self.threshold = threshold
        self.projection_count = projection_count
        self.seed = seed
        self.projection = projection
        self.simhash_bits = simhash_bits
        self.setup: Protocol3PublicSetup | None = None
        self.documents: tuple[Protocol3Document, ...] = ()
        self._secret_to_document: dict[int, Protocol3Document] = {}

    def build_setup_from_embeddings(
        self,
        document_embeddings: torch.Tensor,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Protocol3PublicSetup:
        bits = simhash(document_embeddings, self.projection, self.simhash_bits)
        chunk_values = chunks if chunks is not None else list(range(bits.shape[0]))
        return self.build_setup_from_bits(bits, chunks=chunk_values)

    def build_setup_from_bits(
        self,
        document_bits: torch.Tensor,
        *,
        chunks: Sequence[Any] | None = None,
    ) -> Protocol3PublicSetup:
        if document_bits.dim() != 2:
            raise ValueError("document_bits must have shape [num_docs, simhash_bits]")
        num_docs, bit_length = document_bits.shape
        if chunks is not None and len(chunks) != num_docs:
            raise ValueError("chunks length must match document count")

        masks = protocol3_projection_masks(
            bit_length=bit_length,
            threshold=self.threshold,
            projection_count=self.projection_count,
            seed=self.seed,
        )
        projection_points = protocol3_projection_points(self.projection_count, self.he.modulus, self.seed)
        documents = tuple(
            Protocol3Document(
                index=doc_id,
                simhash_bits=tuple(int(bit) for bit in document_bits[doc_id].reshape(-1).tolist()),
                chunk=chunks[doc_id] if chunks is not None else doc_id,
            )
            for doc_id in range(num_docs)
        )

        entries: dict[bytes, int] = {}
        projection_collision_count = 0
        n = self.he.modulus
        n2_size = _int_byte_size(self.he.modulus_squared)
        rng = random.Random(self.seed + b":shamir")
        self._secret_to_document = {}
        for document in documents:
            secret = _unique_nonzero_mod(n, rng, self._secret_to_document)
            coefficient = _nonzero_mod(n, rng)
            self._secret_to_document[secret] = document
            for mask_id, mask in enumerate(masks):
                key = protocol3_projection_key(document.simhash_bits, mask, mask_id=mask_id)
                share = (coefficient * projection_points[mask_id] + secret) % n
                ciphertext = self.he.encrypt(share)
                if key in entries:
                    projection_collision_count += 1
                    continue
                entries[key] = ciphertext

        if not entries:
            raise ValueError("Protocol 3 requires at least one document")

        keys = []
        values = []
        for key, ciphertext in entries.items():
            keys.append(key)
            values.append(_encode_ciphertext(ciphertext, n2_size))

        table = self.okvs.encode(keys, values)
        setup = Protocol3PublicSetup(
            public_key=self.he.public_key,
            masks=masks,
            projection_points=projection_points,
            table=table,
            simhash_bits=bit_length,
            threshold=self.threshold,
            projection_weight=math.ceil(math.sqrt(self.threshold * bit_length)),
            ciphertext_size=n2_size,
            projection_collision_count=projection_collision_count,
        )
        self.setup = setup
        self.documents = documents
        return setup

    def recover_candidates(self, message: Protocol3ClientMessage) -> tuple[Protocol3Document, ...]:
        candidates: list[Protocol3Document] = []
        seen_indices: set[int] = set()
        for ciphertext in message.shuffled_secret_ciphertexts:
            secret = self.he.decrypt(ciphertext)
            document = self._secret_to_document.get(secret)
            if document is not None and document.index not in seen_indices:
                candidates.append(document)
                seen_indices.add(document.index)
        return tuple(candidates)


class Protocol3Client:
    """Client side of Pisces ∏Oblivious Filter."""

    def __init__(
        self,
        *,
        okvs: BinaryOKVS | None = None,
        projection: torch.Tensor | None = None,
        shuffle_seed: bytes | None = None,
    ) -> None:
        self.okvs = okvs or BinaryOKVS()
        self.projection = projection
        self.shuffle_seed = shuffle_seed
        self.state: Protocol3ClientState | None = None

    def filter_from_embedding(self, query_embedding: torch.Tensor, setup: Protocol3PublicSetup) -> Protocol3ClientMessage:
        bits = simhash(query_embedding, self.projection, setup.simhash_bits).reshape(-1)
        return self.filter_from_bits(bits, setup)

    def filter_from_bits(self, query_bits: torch.Tensor, setup: Protocol3PublicSetup) -> Protocol3ClientMessage:
        if query_bits.numel() != setup.simhash_bits:
            raise ValueError("query_bits length must match setup.simhash_bits")

        n, _ = setup.public_key
        n2 = n * n
        n2_gmp = _gmpy2.mpz(n2) if _gmpy2 is not None else None
        decoded_ciphertexts: list[int] = []
        query_bit_tuple = tuple(int(bit) for bit in query_bits.reshape(-1).tolist())
        for mask_id, mask in enumerate(setup.masks):
            key = protocol3_projection_key(query_bit_tuple, mask, mask_id=mask_id)
            value = self.okvs.decode(setup.table, key)
            decoded_ciphertexts.append(
                _decode_ciphertext(value, ciphertext_size=setup.ciphertext_size, modulus_squared=n2)
            )
        self.state = Protocol3ClientState(tuple(decoded_ciphertexts))

        secret_ciphertexts: list[int] = []
        for left in range(len(decoded_ciphertexts)):
            left_ciphertext = decoded_ciphertexts[left]
            left_x = setup.projection_points[left]
            for right in range(left + 1, len(decoded_ciphertexts)):
                right_x = setup.projection_points[right]
                denominator = (right_x - left_x) % n
                inverse = int(_gmpy2.invert(denominator, n)) if _gmpy2 is not None else pow(denominator, -1, n)
                left_coeff = (right_x * inverse) % n
                right_coeff = (-left_x * inverse) % n
                if _gmpy2 is None:
                    left_term = pow(left_ciphertext, left_coeff, n2)
                    right_term = pow(decoded_ciphertexts[right], right_coeff, n2)
                    secret_ciphertexts.append((left_term * right_term) % n2)
                else:
                    left_term = _gmpy2.powmod(left_ciphertext, left_coeff, n2_gmp)
                    right_term = _gmpy2.powmod(decoded_ciphertexts[right], right_coeff, n2_gmp)
                    secret_ciphertexts.append(int((left_term * right_term) % n2_gmp))

        rng = random.Random(self.shuffle_seed or b"pisces-protocol3-client-shuffle")
        rng.shuffle(secret_ciphertexts)
        return Protocol3ClientMessage(tuple(secret_ciphertexts))


def protocol3_projection_masks(
    *,
    bit_length: int,
    threshold: int,
    projection_count: int,
    seed: bytes,
) -> tuple[tuple[int, ...], ...]:
    if bit_length < 1:
        raise ValueError("bit_length must be positive")
    if threshold < 1:
        raise ValueError("threshold must be positive")
    weight = math.ceil(math.sqrt(threshold * bit_length))
    if weight > bit_length:
        raise ValueError("projection weight cannot exceed bit_length")
    rng = random.Random(seed + b":masks")
    return tuple(tuple(sorted(rng.sample(range(bit_length), weight))) for _ in range(projection_count))


def protocol3_projection_points(projection_count: int, modulus: int, seed: bytes) -> tuple[int, ...]:
    if projection_count < 2:
        raise ValueError("projection_count must be at least 2")
    rng = random.Random(seed + b":projection-points")
    points: set[int] = set()
    while len(points) < projection_count:
        points.add(_nonzero_mod(modulus, rng))
    return tuple(points)


def protocol3_projection_key(bits: Sequence[int], mask: Sequence[int], *, mask_id: int | None = None) -> bytes:
    masked = bytearray(math.ceil(len(bits) / 8))
    for index in mask:
        if bits[index]:
            masked[index // 8] |= 1 << (index % 8)
    mask_domain = b"" if mask_id is None else mask_id.to_bytes(8, "big")
    return hashlib.sha256(b"pisces-protocol3-key" + mask_domain + bytes(masked)).digest()


def protocol3_plain_projection_candidates(
    document_bits: torch.Tensor,
    query_bits: torch.Tensor,
    setup: Protocol3PublicSetup,
    *,
    min_matches: int = 2,
) -> tuple[int, ...]:
    """Plain reference for the ∏Oblivious Filter projected exact-match predicate."""

    if document_bits.dim() != 2:
        raise ValueError("document_bits must have shape [num_docs, simhash_bits]")
    if document_bits.shape[1] != setup.simhash_bits:
        raise ValueError("document_bits width must match setup.simhash_bits")
    if query_bits.numel() != setup.simhash_bits:
        raise ValueError("query_bits length must match setup.simhash_bits")

    query_tuple = tuple(int(bit) for bit in query_bits.reshape(-1).tolist())
    candidates: list[int] = []
    for doc_id in range(document_bits.shape[0]):
        doc_tuple = tuple(int(bit) for bit in document_bits[doc_id].reshape(-1).tolist())
        matches = 0
        for mask_id, mask in enumerate(setup.masks):
            if protocol3_projection_key(doc_tuple, mask, mask_id=mask_id) == protocol3_projection_key(
                query_tuple,
                mask,
                mask_id=mask_id,
            ):
                matches += 1
        if matches >= min_matches:
            candidates.append(doc_id)
    return tuple(candidates)


def paillier_encrypt(plaintext: int, public_key: tuple[int, int], rng: random.Random | random.SystemRandom) -> int:
    n, g = public_key
    n2 = n * n
    message = plaintext % n
    while True:
        r = rng.randrange(1, n)
        if math.gcd(r, n) == 1:
            break
    return (pow(g, message, n2) * pow(r, n, n2)) % n2


def paillier_add(left: int, right: int, public_key: tuple[int, int]) -> int:
    n = public_key[0]
    return (left * right) % (n * n)


def paillier_mul_plain(ciphertext: int, scalar: int, public_key: tuple[int, int]) -> int:
    n = public_key[0]
    n2 = n * n
    if _gmpy2 is not None:
        return int(_gmpy2.powmod(ciphertext, scalar % n, n2))
    return pow(ciphertext, scalar % n, n2)


def paillier_interpolate_at_zero(
    left_ciphertext: int,
    left_x: int,
    right_ciphertext: int,
    right_x: int,
    public_key: tuple[int, int],
) -> int:
    n = public_key[0]
    denominator = (right_x - left_x) % n
    inverse = int(_gmpy2.invert(denominator, n)) if _gmpy2 is not None else pow(denominator, -1, n)
    left_coeff = (right_x * inverse) % n
    right_coeff = (-left_x * inverse) % n
    return paillier_add(
        paillier_mul_plain(left_ciphertext, left_coeff, public_key),
        paillier_mul_plain(right_ciphertext, right_coeff, public_key),
        public_key,
    )


def _encode_ciphertext(ciphertext: int, ciphertext_size: int) -> bytes:
    return ciphertext.to_bytes(ciphertext_size, "big")


def _decode_ciphertext(
    value: bytes,
    *,
    ciphertext_size: int,
    modulus_squared: int,
) -> int:
    if len(value) != ciphertext_size:
        raise ValueError(f"ciphertext value has size {len(value)}, expected {ciphertext_size}")
    return int.from_bytes(value, "big") % modulus_squared


def run_protocol3_server(
    party,
    protocol: Protocol3Server,
    document_bits: torch.Tensor,
    *,
    chunks: Sequence[Any] | None = None,
) -> tuple[Protocol3Document, ...]:
    """Run ∏Oblivious Filter on a server party.

    Message order:
    1. server -> client: public HE key, projection masks, projection points, OKVS
    2. client -> server: shuffled encrypted candidate secrets
    """

    setup = protocol.build_setup_from_bits(document_bits, chunks=chunks)
    party.send(setup)
    message = party.receive()
    return protocol.recover_candidates(message)


def run_protocol3_client(party, protocol: Protocol3Client, query_bits: torch.Tensor) -> Protocol3ClientMessage:
    """Run ∏Oblivious Filter on a client party and send shuffled encrypted secrets."""

    setup = party.receive()
    message = protocol.filter_from_bits(query_bits, setup)
    party.send(message)
    return message


def _int_byte_size(value: int) -> int:
    return max(1, (value.bit_length() + 7) // 8)


def _nonzero_mod(modulus: int, rng: random.Random) -> int:
    return rng.randrange(1, modulus)


def _unique_nonzero_mod(modulus: int, rng: random.Random, existing: dict[int, Any]) -> int:
    while True:
        value = _nonzero_mod(modulus, rng)
        if value not in existing:
            return value
