import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig, PiscesRetriever
from NssMPC.application.rag.pisces.ops import bm25_components_from_tf, bm25_scores_from_tf
from NssMPC.application.rag.pisces.protocol4 import (
    Protocol4Client,
    Protocol4Server,
    protocol4_label_key,
    protocol4_okvs_key,
)
from NssMPC.application.rag.pisces.psi import OKVSMultiInstanceLabeledPSI
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer, OPRFBlindRequest, OPRFBlindResponse


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def int_prefix(value, chars=34):
    return hex(value)[:chars]


def bytes_prefix(value, size=12):
    return value[:size].hex()


def test_binary_okvs_roundtrip():
    section("1. Binary OKVS roundtrip")
    okvs = BinaryOKVS(expansion=2.4, seed=b"okvs-test-seed")
    keys = [f"key-{i}".encode() for i in range(64)]
    values = [i.to_bytes(8, "big") + bytes([i]) * 8 for i in range(64)]
    print("[Input] keys: 64 fixed keys, values: 64 fixed 16-byte values")
    print("[Expected] For every inserted key k_i, Decode(Encode(k_i -> v_i), k_i) == v_i")
    table = okvs.encode(keys, values)
    print(
        f"[Actual] OKVS slots={table.size}, value_size={table.value_size}, "
        f"method={table.method}, seed_prefix={bytes_prefix(table.seed)}"
    )

    for key, value in zip(keys, values):
        assert okvs.decode(table, key) == value
    print("[Check] all 64 inserted keys decode to their original values")


def test_protocol4_term_frequencies_match_plaintext():
    section("2. Local OKVS-backed Protocol 4 TF recovery")
    tf = torch.zeros(16, 5)
    tf[1, 0] = 2
    tf[1, 3] = 1
    tf[4, 2] = 3
    tf[7, 0] = 1
    tf[7, 4] = 5
    tf[9, 1] = 4

    query = torch.tensor([1, 7, 8, 9])
    print("[Input] document_term_frequency shape=[vocab_size=16, num_docs=5]")
    print(tf)
    print(f"[Input] query tokens={query.tolist()}")
    print("[Expected] Client should recover tf[query].T, shape=[num_docs, query_terms].")
    psi = OKVSMultiInstanceLabeledPSI(okvs=BinaryOKVS(expansion=2.4, seed=b"psi-okvs-test"), oprf_key=b"oprf-test-key")
    got = psi.term_frequencies(query, tf)
    expected = tf[query].transpose(0, 1).float()
    print(f"[Actual] OKVS slots={psi.table.size}, value_size={psi.table.value_size}, num_docs={psi.num_docs}")
    print("[Expected TF]")
    print(expected)
    print("[Recovered TF]")
    print(got)

    assert torch.equal(got, expected)
    print("[Check] recovered TF exactly matches plaintext baseline")


def test_binary_okvs_linear_core_roundtrip():
    section("2a. Binary OKVS 2-core linear solve")
    okvs = BinaryOKVS(expansion=1.05, seed=b"force-linear")
    keys = [f"k-{i}".encode() for i in range(80)]
    values = [i.to_bytes(8, "big") + bytes([255 - i]) * 8 for i in range(80)]
    print("[Input] 80 keys with low expansion=1.05, which is expected to create a non-empty 2-core.")
    print("[Expected] Encode should fall back from peeling to GF(2) linear solving and still decode every inserted key.")
    table = okvs.encode(keys, values)
    print(
        f"[Actual] OKVS slots={table.size}, value_size={table.value_size}, "
        f"method={table.method}, seed_prefix={bytes_prefix(table.seed)}"
    )

    assert table.method == "linear"
    for key, value in zip(keys, values):
        assert okvs.decode(table, key) == value
    print("[Check] linear-solved OKVS decodes all inserted values exactly")


def test_dh_oprf_matches_server_direct_evaluation():
    section("3. DH-OPRF blind/evaluate/finalize")
    params = DHOPRFParams()
    server = DHOPRFServer(params=params, secret_key=123456789)
    client = DHOPRFClient(params=params)
    items = [1, 7, 42]
    print(f"[Input] items={items}")
    print("[Expected] Client finalizes server evaluation of blinded HashToGroup(item) to the same PRF output as server direct evaluation.")

    request, state = client.blind(items)
    for item, blind, blinded in zip(items, state.blinds, request.elements):
        point = params.hash_to_group(item)
        print(
            f"[Blind] item={item}, H(item)={int_prefix(point)}, "
            f"blind={int_prefix(blind)}, blinded={int_prefix(blinded)}"
        )
    response = server.evaluate(request)
    for item, evaluated in zip(items, response.elements):
        print(f"[Server evaluate] item={item}, evaluated_blinded={int_prefix(evaluated)}")
    got = client.finalize(state, response)
    expected = [server.evaluate_direct(item) for item in items]
    for item, actual, expect in zip(items, got, expected):
        print(f"[Finalize] item={item}, client_output={bytes_prefix(actual)}, expected_direct={bytes_prefix(expect)}")

    assert got == expected
    print("[Check] all finalized OPRF outputs match server direct PRF outputs")


def test_dh_oprf_rejects_invalid_group_elements():
    section("3a. DH-OPRF group-element validation")
    params = DHOPRFParams()
    server = DHOPRFServer(params=params, secret_key=123456789)
    client = DHOPRFClient(params=params)
    print("[Input] invalid OPRF group elements: 1 and p - 1")
    print("[Expected] Server rejects invalid blinded requests and client rejects invalid blinded responses.")

    try:
        server.evaluate(OPRFBlindRequest((1,)))
    except ValueError as exc:
        print(f"[Server reject] {exc}")
    else:
        raise AssertionError("server accepted an invalid OPRF request element")

    request, state = client.blind([1])
    try:
        client.finalize(state, OPRFBlindResponse((params.p - 1,)))
    except ValueError as exc:
        print(f"[Client reject] {exc}")
    else:
        raise AssertionError("client accepted an invalid OPRF response element")

    assert params.is_valid_group_element(request.elements[0])
    print("[Check] valid blinded request element remains accepted by the subgroup predicate")


def test_interactive_protocol4_term_frequencies_match_plaintext():
    section("4. Interactive Protocol 4 server/client API")
    tf = torch.zeros(18, 6)
    tf[1, 0] = 2
    tf[1, 3] = 1
    tf[4, 2] = 3
    tf[7, 0] = 1
    tf[7, 4] = 5
    tf[9, 1] = 4
    tf[12, 5] = 6

    query = torch.tensor([1, 7, 8, 12])
    print("[Input] document_term_frequency shape=[vocab_size=18, num_docs=6]")
    print(tf)
    print(f"[Input] query tokens={query.tolist()}")
    print("[Expected] Protocol 4 returns only the query-token TF columns, arranged as [num_docs, query_terms].")
    params = DHOPRFParams()
    server = Protocol4Server(
        okvs=BinaryOKVS(expansion=2.4, seed=b"interactive-protocol4-okvs"),
        oprf_server=DHOPRFServer(params=params, secret_key=987654321),
    )
    client = Protocol4Client(
        okvs=server.okvs,
        oprf_client=DHOPRFClient(params=params),
    )

    setup = server.build_setup(tf)
    nonzero_pairs = int((tf > 0).sum().item())
    print(
        "[Server setup] Expected one OKVS label per nonzero (token, doc) TF entry. "
        f"nonzero_pairs={nonzero_pairs}"
    )
    print(
        f"[Server setup] OKVS slots={setup.table.size}, value_size={setup.table.value_size}, "
        f"method={setup.table.method}, num_docs={setup.num_docs}, seed_prefix={bytes_prefix(setup.table.seed)}"
    )
    request = client.make_query(query)
    print(f"[Client request] Expected one blinded OPRF element per query token. count={len(request.elements)}")
    for token, element in zip(query.tolist(), request.elements):
        print(f"[Client request] token={token}, blinded_element={int_prefix(element)}")
    response = server.evaluate_oprf(request)
    print(f"[Server response] Expected same count of evaluated blinded elements. count={len(response.elements)}")
    for token, element in zip(query.tolist(), response.elements):
        print(f"[Server response] token={token}, evaluated_blinded={int_prefix(element)}")
    got = client.recover_term_frequencies(response, setup)
    expected = tf[query].transpose(0, 1).float()
    print("[Expected TF]")
    print(expected)
    print("[Recovered TF]")
    print(got)

    token = int(query[0].item())
    doc_id = int(torch.nonzero(tf[token] > 0, as_tuple=False)[0].item())
    sample_prf_values = client.oprf_client.finalize(client._blind_state, response)
    client_prf_value = sample_prf_values[0]
    server_prf_value = server.oprf_server.evaluate_direct(token)
    okvs_key = protocol4_okvs_key(doc_id, client_prf_value)
    label_key = protocol4_label_key(doc_id, client_prf_value)
    ciphertext = server.okvs.decode(setup.table, okvs_key)
    payload = server.cipher.decrypt(label_key, ciphertext)
    expected_prefix = b"\x00" * setup.prefix_size
    expected_tf = int(tf[token, doc_id].item())
    decoded_tf = int.from_bytes(payload[setup.prefix_size : setup.prefix_size + setup.tf_size], "big")
    print(
        "[Decode sample] This sample should decrypt to 0^prefix || tf for a real matching token/doc pair: "
        f"token={token}, doc_id={doc_id}, client_prf_prefix={bytes_prefix(client_prf_value)}, "
        f"server_direct_prf_prefix={bytes_prefix(server_prf_value)}, "
        f"client_equals_server_direct={client_prf_value == server_prf_value}, "
        f"okvs_key_prefix={bytes_prefix(okvs_key)}, "
        f"cipher_prefix={bytes_prefix(ciphertext)}, prefix_ok={payload[:setup.prefix_size] == expected_prefix}, "
        f"decoded_tf={decoded_tf}, expected_tf={expected_tf}"
    )

    assert torch.equal(got, expected)
    print("[Check] interactive Protocol 4 recovered TF exactly matches plaintext baseline")


def test_retriever_uses_okvs_protocol4_for_plain_bm25():
    section("5. PiscesRetriever lexical path consumes Protocol 4-style TF")
    tf = torch.zeros(20, 6)
    tf[2, 0] = 3
    tf[2, 1] = 1
    tf[5, 4] = 2
    tf[11, 2] = 5
    tf[11, 5] = 1
    query = torch.tensor([2, 11])
    lengths = torch.tensor([8, 7, 9, 6, 10, 11]).float()
    payload = torch.arange(6 * 3).reshape(6, 3).float()

    labeled_psi = OKVSMultiInstanceLabeledPSI(
        okvs=BinaryOKVS(expansion=2.4, seed=b"retriever-okvs-test"),
        oprf_key=b"retriever-oprf-test-key",
    )
    retriever = PiscesRetriever(PiscesConfig(top_k=2), labeled_psi=labeled_psi)
    print(f"[Input] query tokens={query.tolist()}, document lengths={lengths.tolist()}")
    print("[Expected] Retriever asks labeled_psi for TF, computes BM25 scores, then Top-K indicators select payload rows.")
    scores, indicators, docs = retriever.lexical_path_plain(query, tf, lengths, document_payload=payload)
    expected_tf = tf[query].transpose(0, 1).float()
    print("[Expected TF from Protocol 4 boundary]")
    print(expected_tf)
    print(f"[Actual] scores shape={tuple(scores.shape)}, indicators shape={tuple(indicators.shape)}, docs shape={tuple(docs.shape)}")
    print("[Actual] BM25 scores")
    print(scores)
    print("[Actual] Top-K indicators")
    print(indicators)
    print("[Actual] selected payload docs")
    print(docs)

    assert scores.shape == (6,)
    assert indicators.shape == (2, 6)
    assert docs.shape == (2, 3)
    print("[Check] retriever output shapes are correct")


def test_protocol2_bm25_components_match_score_helper():
    section("6. Protocol 2 BM25 component audit")
    tf = torch.tensor(
        [
            [3.0, 0.0],
            [1.0, 0.0],
            [0.0, 5.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    lengths = torch.tensor([8.0, 7.0, 9.0, 6.0, 11.0])
    avg_len = float(lengths.mean().item())
    print("[Input] recovered TF [num_docs, query_terms]")
    print(tf)
    print(f"[Input] document lengths={lengths.tolist()}, average_length={avg_len:.6f}")
    print("[Expected] Protocol 2 computes df, idf, length normalization, per-token contributions, then sums BM25 scores.")
    bm25 = bm25_components_from_tf(tf, lengths, num_documents=tf.shape[0], average_document_length=avg_len)
    scores = bm25_scores_from_tf(tf, lengths, num_documents=tf.shape[0], average_document_length=avg_len)
    print(f"[Actual] df={bm25.df.tolist()}")
    print(f"[Actual] idf={bm25.idf.tolist()}")
    print(f"[Actual] length_norm={bm25.length_norm.tolist()}")
    print("[Actual] contributions")
    print(bm25.contributions)
    print(f"[Actual] scores={bm25.scores.tolist()}")

    assert torch.allclose(bm25.scores, scores)
    print("[Check] BM25 component helper matches score helper exactly")


def main():
    test_binary_okvs_roundtrip()
    test_protocol4_term_frequencies_match_plaintext()
    test_binary_okvs_linear_core_roundtrip()
    test_dh_oprf_matches_server_direct_evaluation()
    test_dh_oprf_rejects_invalid_group_elements()
    test_interactive_protocol4_term_frequencies_match_plaintext()
    test_retriever_uses_okvs_protocol4_for_plain_bm25()
    test_protocol2_bm25_components_match_score_helper()
    print("pisces protocol4 tests ok")


if __name__ == "__main__":
    main()
