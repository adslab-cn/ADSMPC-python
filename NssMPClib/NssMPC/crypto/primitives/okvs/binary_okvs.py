"""Binary OKVS for Pisces Protocol 4.

This module implements a 3-hash binary OKVS over GF(2). Values are fixed-length
byte strings and Decode XORs the three table locations selected by the key.
Encode first tries the usual peeling assignment; if the cuckoo hypergraph has a
2-core, it solves the remaining linear system over GF(2) instead of giving up.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from collections import deque
from typing import Iterable, Sequence


class OKVSEncodeError(RuntimeError):
    """Raised when all OKVS encode retries fail."""


@dataclass(frozen=True)
class OKVSTable:
    slots: tuple[bytes, ...]
    seed: bytes
    value_size: int
    method: str = "unknown"

    @property
    def size(self) -> int:
        return len(self.slots)


def xor_bytes(*values: bytes) -> bytes:
    if not values:
        return b""
    size = len(values[0])
    out = bytearray(size)
    for value in values:
        if len(value) != size:
            raise ValueError("all byte strings must have the same length")
        for i, byte in enumerate(value):
            out[i] ^= byte
    return bytes(out)


class BinaryOKVS:
    """3-hash binary OKVS over fixed-length byte strings."""

    def __init__(
        self,
        *,
        expansion: float = 2.0,
        hash_count: int = 3,
        max_retries: int = 128,
        seed: bytes | None = None,
    ) -> None:
        if hash_count != 3:
            raise ValueError("this implementation currently supports exactly 3 hash functions")
        if expansion <= 1.0:
            raise ValueError("expansion must be greater than 1")
        self.expansion = expansion
        self.hash_count = hash_count
        self.max_retries = max_retries
        self.seed = seed or os.urandom(32)

    def encode(self, keys: Sequence[bytes], values: Sequence[bytes]) -> OKVSTable:
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")
        if not keys:
            return OKVSTable((), self.seed, 0)
        if len(set(keys)) != len(keys):
            raise ValueError("OKVS keys must be distinct")

        value_size = len(values[0])
        if value_size == 0:
            raise ValueError("OKVS values must be non-empty fixed-length bytes")
        if any(len(value) != value_size for value in values):
            raise ValueError("all OKVS values must have the same length")

        table_size = max(self.hash_count + 1, math.ceil(len(keys) * self.expansion))
        last_error = None
        for retry in range(self.max_retries):
            retry_seed = self._retry_seed(retry)
            edges = [self._slots_for_key(key, table_size, retry_seed) for key in keys]
            try:
                slots = self._peel_and_assign(edges, values, table_size, value_size)
                return OKVSTable(tuple(slots), retry_seed, value_size, "peeling")
            except OKVSEncodeError as exc:
                last_error = exc
                try:
                    slots = self._solve_linear(edges, values, table_size, value_size)
                    return OKVSTable(tuple(slots), retry_seed, value_size, "linear")
                except OKVSEncodeError as linear_exc:
                    last_error = linear_exc
        raise OKVSEncodeError(f"failed to encode {len(keys)} keys after {self.max_retries} retries") from last_error

    def decode(self, table: OKVSTable, key: bytes) -> bytes:
        if table.value_size == 0:
            return b""
        edge = self._slots_for_key(key, table.size, table.seed)
        return xor_bytes(*(table.slots[index] for index in edge))

    def _retry_seed(self, retry: int) -> bytes:
        return hashlib.sha256(self.seed + retry.to_bytes(8, "big")).digest()

    def _slots_for_key(self, key: bytes, table_size: int, seed: bytes) -> tuple[int, int, int]:
        slots: list[int] = []
        counter = 0
        while len(slots) < self.hash_count:
            digest = hashlib.sha512(seed + key + counter.to_bytes(4, "big")).digest()
            for offset in range(0, len(digest), 8):
                candidate = int.from_bytes(digest[offset : offset + 8], "big") % table_size
                if candidate not in slots:
                    slots.append(candidate)
                    if len(slots) == self.hash_count:
                        break
            counter += 1
        return slots[0], slots[1], slots[2]

    def _peel_and_assign(
        self,
        edges: Sequence[tuple[int, int, int]],
        values: Sequence[bytes],
        table_size: int,
        value_size: int,
    ) -> list[bytes]:
        incident: list[set[int]] = [set() for _ in range(table_size)]
        for edge_id, edge in enumerate(edges):
            for node in edge:
                incident[node].add(edge_id)

        degrees = [len(edge_ids) for edge_ids in incident]
        queue = deque(node for node, degree in enumerate(degrees) if degree == 1)
        removed = [False] * len(edges)
        stack: list[tuple[int, int]] = []

        while queue:
            node = queue.popleft()
            if degrees[node] != 1:
                continue
            edge_id = next((eid for eid in incident[node] if not removed[eid]), None)
            if edge_id is None:
                continue
            removed[edge_id] = True
            stack.append((edge_id, node))
            for edge_node in edges[edge_id]:
                degrees[edge_node] -= 1
                if degrees[edge_node] == 1:
                    queue.append(edge_node)

        if len(stack) != len(edges):
            raise OKVSEncodeError("peeling failed; hypergraph has a non-empty 2-core")

        table = [os.urandom(value_size) for _ in range(table_size)]
        for edge_id, target_node in reversed(stack):
            known = [table[node] for node in edges[edge_id] if node != target_node]
            table[target_node] = xor_bytes(values[edge_id], *known)
        return table

    def _solve_linear(
        self,
        edges: Sequence[tuple[int, int, int]],
        values: Sequence[bytes],
        table_size: int,
        value_size: int,
    ) -> list[bytes]:
        rows = []
        for edge, value in zip(edges, values):
            mask = 0
            for node in edge:
                mask ^= 1 << node
            rows.append([mask, value])

        pivot_columns: list[int] = []
        row = 0
        for column in range(table_size):
            pivot = None
            for candidate in range(row, len(rows)):
                if (rows[candidate][0] >> column) & 1:
                    pivot = candidate
                    break
            if pivot is None:
                continue

            rows[row], rows[pivot] = rows[pivot], rows[row]
            for other in range(len(rows)):
                if other != row and ((rows[other][0] >> column) & 1):
                    rows[other][0] ^= rows[row][0]
                    rows[other][1] = xor_bytes(rows[other][1], rows[row][1])
            pivot_columns.append(column)
            row += 1

        for mask, value in rows[row:]:
            if mask == 0 and value != b"\x00" * value_size:
                raise OKVSEncodeError("linear OKVS system is inconsistent")

        pivot_column_set = set(pivot_columns)
        table = [os.urandom(value_size) for _ in range(table_size)]
        for column in pivot_column_set:
            table[column] = b"\x00" * value_size

        pivot_rows = [(column, rows[index][0], rows[index][1]) for index, column in enumerate(pivot_columns)]
        for column, mask, value in reversed(pivot_rows):
            assigned = [table[index] for index in range(table_size) if index != column and ((mask >> index) & 1)]
            table[column] = xor_bytes(value, *assigned)
        return table
