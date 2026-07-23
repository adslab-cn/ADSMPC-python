import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces.protocol3 import (
    AdditivePaillier,
    Protocol3Client,
    Protocol3Server,
    paillier_interpolate_at_zero,
    protocol3_plain_projection_candidates,
)
from NssMPC.crypto.primitives.okvs import BinaryOKVS


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_paillier_interpolation_recovers_shamir_secret():
    section("1. Protocol 3 Paillier interpolation")
    he = AdditivePaillier(key_size=64)
    n = he.modulus
    secret = 12345 % n
    coefficient = 6789 % n
    x1 = 11
    x2 = 29
    y1 = (coefficient * x1 + secret) % n
    y2 = (coefficient * x2 + secret) % n
    c1 = he.encrypt(y1)
    c2 = he.encrypt(y2)
    encrypted_secret = paillier_interpolate_at_zero(c1, x1, c2, x2, he.public_key)
    recovered = he.decrypt(encrypted_secret)

    print(f"[Input] secret={secret}, coefficient={coefficient}, points=({x1}, {y1}), ({x2}, {y2})")
    print("[Expected] Homomorphic interpolation at x=0 decrypts to the original Shamir secret.")
    print(f"[Actual] recovered={recovered}")
    assert recovered == secret
    print("[Check] encrypted interpolation recovers the secret exactly")


def test_protocol3_oblivious_filter_exact_match_candidate():
    section("2. Protocol 3 OKVS-backed oblivious filter")
    document_bits = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1, 0, 0] * 4,
            [0, 1, 0, 0, 1, 0, 1, 1] * 4,
            [1, 1, 0, 1, 0, 0, 1, 0] * 4,
        ],
        dtype=torch.uint8,
    )
    query_bits = document_bits[0].clone()
    chunks = ["doc-0", "doc-1", "doc-2"]
    server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"protocol3-okvs"),
        he=AdditivePaillier(key_size=64),
        threshold=2,
        projection_count=8,
        seed=b"protocol3-test",
    )
    client = Protocol3Client(okvs=server.okvs, shuffle_seed=b"protocol3-client-test")

    print(f"[Input] document_bits shape={tuple(document_bits.shape)}, query equals doc index 0")
    print("[Expected] The client reconstructs encrypted Shamir secrets only for matching projections; server learns doc-0 as a candidate.")
    setup = server.build_setup_from_bits(document_bits, chunks=chunks)
    print(
        f"[Setup] masks={len(setup.masks)}, projection_weight={setup.projection_weight}, "
        f"OKVS slots={setup.table.size}, "
        f"value_size={setup.table.value_size}, method={setup.table.method}"
    )
    message = client.filter_from_bits(query_bits, setup)
    print(f"[Client] decoded={len(client.state.decoded_ciphertexts)}, shuffled pair ciphertexts={len(message.shuffled_secret_ciphertexts)}")
    candidates = server.recover_candidates(message)
    candidate_chunks = [candidate.chunk for candidate in candidates]
    print(f"[Server] candidates={candidate_chunks}")

    assert "doc-0" in candidate_chunks
    assert len(candidate_chunks) >= 1
    print("[Check] exact matching document appears in Protocol 3 candidate set")


def test_protocol3_uses_one_ciphertext_per_projection():
    section("3. Protocol 3 single-value projection encoding")
    document_bits = torch.tensor(
        [
            [1, 0, 1, 1, 0, 1, 0, 0] * 4,
            [1, 0, 1, 1, 0, 1, 0, 0] * 4,
            [0, 1, 0, 0, 1, 0, 1, 1] * 4,
        ],
        dtype=torch.uint8,
    )
    query_bits = document_bits[0].clone()
    chunks = ["doc-0", "doc-1", "doc-2"]
    server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"protocol3-single-okvs"),
        he=AdditivePaillier(key_size=64),
        threshold=2,
        projection_count=8,
        seed=b"protocol3-single-test",
    )
    client = Protocol3Client(okvs=server.okvs, shuffle_seed=b"protocol3-single-client-test")

    print("[Input] doc-0 and doc-1 collide on every projection key.")
    print("[Expected] Protocol 3 stores one ciphertext per projection key, so client sends C(8,2)=28 interpolations.")
    setup = server.build_setup_from_bits(document_bits, chunks=chunks)
    message = client.filter_from_bits(query_bits, setup)
    candidates = server.recover_candidates(message)
    candidate_chunks = [candidate.chunk for candidate in candidates]
    expected_ciphertexts = len(setup.masks) * (len(setup.masks) - 1) // 2
    print(
        f"[Setup] projection_collision_count={setup.projection_collision_count}, "
        f"decoded_projection_ciphertexts={len(client.state.decoded_ciphertexts)}"
    )
    print(f"[Client] encrypted interpolation count={len(message.shuffled_secret_ciphertexts)}")
    print(f"[Server] candidates={candidate_chunks}")

    assert setup.projection_collision_count > 0
    assert len(message.shuffled_secret_ciphertexts) == expected_ciphertexts
    assert len(candidate_chunks) == 1
    print("[Check] single-value projection encoding reduces combinations to projection pairs")


if __name__ == "__main__":
    test_paillier_interpolation_recovers_shamir_secret()
    test_protocol3_oblivious_filter_exact_match_candidate()
    test_protocol3_uses_one_ciphertext_per_projection()
    print("pisces protocol3 tests ok")
