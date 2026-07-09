"""Finite-field Diffie-Hellman OPRF.

This is a real two-party OPRF backend for prototypes: the client blinds
HashToGroup(x), the server exponentiates with its secret key, and the client
unblinds to obtain Hash(HashToGroup(x)^k). It follows the classic blind
exponentiation logic used by DDH-style OPRFs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
from typing import Iterable, Sequence


RFC3526_2048_PRIME_HEX = """
FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
E485B576 625E7EC6 F44C42E9 A637ED6B 0BFF5CB6 F406B7ED
EE386BFB 5A899FA5 AE9F2411 7C4B1FE6 49286651 ECE45B3D
C2007CB8 A163BF05 98DA4836 1C55D39A 69163FA8 FD24CF5F
83655D23 DCA3AD96 1C62F356 208552BB 9ED52907 7096966D
670C354E 4ABC9804 F1746C08 CA18217C 32905E46 2E36CE3B
E39E772C 180E8603 9B2783A2 EC07A28F B5C55DF0 6F4C52C9
DE2BCBF6 95581718 3995497C EA956AE5 15D22618 98FA0510
15728E5A 8AACAA68 FFFFFFFF FFFFFFFF
"""


def encode_item(item: int | str | bytes) -> bytes:
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return int(item).to_bytes(16, "big", signed=True)


@dataclass(frozen=True)
class DHOPRFParams:
    p: int = int("".join(RFC3526_2048_PRIME_HEX.split()), 16)
    domain: bytes = b"nssmpc-pisces-dh-oprf-v1"

    @property
    def q(self) -> int:
        return (self.p - 1) // 2

    @property
    def element_size(self) -> int:
        return (self.p.bit_length() + 7) // 8

    def hash_to_group(self, item: int | str | bytes) -> int:
        seed = self.domain + b":hash-to-group:" + encode_item(item)
        counter = 0
        while True:
            digest = hashlib.sha512(seed + counter.to_bytes(4, "big")).digest()
            candidate = int.from_bytes(digest, "big") % self.p
            if candidate not in (0, 1, self.p - 1):
                point = pow(candidate, 2, self.p)
                if self.is_valid_group_element(point):
                    return point
            counter += 1

    def is_valid_group_element(self, element: int) -> bool:
        return 1 < element < self.p - 1 and pow(element, self.q, self.p) == 1

    def require_valid_group_element(self, element: int) -> None:
        if not self.is_valid_group_element(element):
            raise ValueError("OPRF group element is not in the expected prime-order subgroup")

    def finalize(self, item: int | str | bytes, group_element: int) -> bytes:
        self.require_valid_group_element(group_element)
        element_bytes = group_element.to_bytes(self.element_size, "big")
        return hashlib.sha256(self.domain + b":finalize:" + encode_item(item) + element_bytes).digest()


@dataclass(frozen=True)
class OPRFBlindRequest:
    elements: tuple[int, ...]


@dataclass(frozen=True)
class OPRFBlindResponse:
    elements: tuple[int, ...]


@dataclass(frozen=True)
class OPRFBlindState:
    items: tuple[int | str | bytes, ...]
    blinds: tuple[int, ...]


class DHOPRFServer:
    def __init__(self, *, params: DHOPRFParams | None = None, secret_key: int | None = None) -> None:
        self.params = params or DHOPRFParams()
        self.secret_key = secret_key or self._random_scalar()
        if not 1 <= self.secret_key < self.params.q:
            raise ValueError("secret_key must be in [1, q)")

    def evaluate(self, request: OPRFBlindRequest) -> OPRFBlindResponse:
        for element in request.elements:
            self.params.require_valid_group_element(element)
        return OPRFBlindResponse(tuple(pow(element, self.secret_key, self.params.p) for element in request.elements))

    def evaluate_direct(self, item: int | str | bytes) -> bytes:
        point = self.params.hash_to_group(item)
        evaluated = pow(point, self.secret_key, self.params.p)
        return self.params.finalize(item, evaluated)

    def _random_scalar(self) -> int:
        return secrets.randbelow(self.params.q - 1) + 1


class DHOPRFClient:
    def __init__(self, *, params: DHOPRFParams | None = None) -> None:
        self.params = params or DHOPRFParams()

    def blind(self, items: Sequence[int | str | bytes]) -> tuple[OPRFBlindRequest, OPRFBlindState]:
        blinds = tuple(self._random_scalar() for _ in items)
        elements = []
        for item, blind in zip(items, blinds):
            point = self.params.hash_to_group(item)
            elements.append(pow(point, blind, self.params.p))
        return OPRFBlindRequest(tuple(elements)), OPRFBlindState(tuple(items), blinds)

    def finalize(self, state: OPRFBlindState, response: OPRFBlindResponse) -> list[bytes]:
        if len(state.items) != len(response.elements):
            raise ValueError("OPRF response length does not match blind state")

        outputs = []
        for item, blind, evaluated in zip(state.items, state.blinds, response.elements):
            self.params.require_valid_group_element(evaluated)
            inverse = pow(blind, -1, self.params.q)
            unblinded = pow(evaluated, inverse, self.params.p)
            outputs.append(self.params.finalize(item, unblinded))
        return outputs

    def _random_scalar(self) -> int:
        return secrets.randbelow(self.params.q - 1) + 1
