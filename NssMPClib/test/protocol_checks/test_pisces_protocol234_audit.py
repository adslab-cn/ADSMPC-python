import itertools
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC import ArithmeticSecretSharing
from NssMPC.application.neural_network.utils.converter import share_data
from NssMPC.application.rag.pisces.config import PiscesConfig
from NssMPC.application.rag.pisces.ops import bm25_components_from_tf, default_average_length
from NssMPC.application.rag.pisces.protocol3 import (
    AdditivePaillier,
    Protocol3Client,
    Protocol3ClientMessage,
    Protocol3Server,
    paillier_interpolate_at_zero,
    protocol3_plain_projection_candidates,
    protocol3_projection_key,
)
from NssMPC.application.rag.pisces.protocol4 import (
    Protocol4Client,
    Protocol4Server,
    protocol4_label_key,
    protocol4_okvs_key,
)
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


def section(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def short_hex(data, size=18):
    if isinstance(data, int):
        return hex(data)[: 2 + size]
    return data.hex()[:size]


def restore(share0, share1):
    return ArithmeticSecretSharing.restore_from_shares(share0, share1).convert_to_real_field()


def assert_close(name, actual, expected, *, atol=1e-3):
    max_diff = (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0
    print(f"[Check] {name}: max_diff={max_diff:.6g}")
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=atol):
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}")


def audit_protocol4_dense_labeled_psi():
    section("Protocol 4 audit: OPRF + OKVS + AES label encryption")

    tf_by_token_doc = torch.tensor(
        [
            [0, 0, 0, 0],
            [3, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 4],
            [5, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    query_tokens = torch.tensor([1, 3, 4, 5], dtype=torch.long)
    expected_tf = tf_by_token_doc[query_tokens].T.contiguous()
    print("[Plain input] server TF matrix [vocab, docs]:")
    print(tf_by_token_doc)
    print(f"[Plain input] client query tokens={query_tokens.tolist()}")
    print("[Plain expected] TF(query_tokens) transposed to [docs, query_terms]:")
    print(expected_tf)

    params = DHOPRFParams()
    okvs = BinaryOKVS(expansion=2.4)
    server = Protocol4Server(okvs=okvs, oprf_server=DHOPRFServer(params=params, secret_key=13579))
    client = Protocol4Client(okvs=okvs, oprf_client=DHOPRFClient(params=params))

    setup = server.build_setup(tf_by_token_doc)
    request = client.make_query(query_tokens)
    response = server.evaluate_oprf(request)
    finalized_prfs = client.oprf_client.finalize(client._blind_state, response)
    server_direct_prfs = [server.oprf_server.evaluate_direct(int(token.item())) for token in query_tokens]

    print(f"[Public setup] num_docs={setup.num_docs}, okvs_slots={setup.table.size}, label_bytes={setup.prefix_size + setup.tf_size}")
    print(f"[Ciphertext check] blinded OPRF request count={len(request.elements)}")
    print(f"[Ciphertext check] first query token plaintext={int(query_tokens[0])}")
    print(f"[Ciphertext check] first blinded element={short_hex(request.elements[0])} (not the token)")
    print(f"[Ciphertext check] first blinded response={short_hex(response.elements[0])}")
    print(f"[Decrypt/audit] first finalized PRF prefix={short_hex(finalized_prfs[0])}")
    print(f"[Plain recompute] first server direct PRF prefix={short_hex(server_direct_prfs[0])}")
    assert finalized_prfs == server_direct_prfs

    token = int(query_tokens[0].item())
    doc_id = 0
    prf_value = server_direct_prfs[0]
    okvs_key = protocol4_okvs_key(doc_id, prf_value)
    label_key = protocol4_label_key(doc_id, prf_value)
    ciphertext = okvs.decode(setup.table, okvs_key)
    payload = server.cipher.decrypt(label_key, ciphertext)
    expected_payload = (b"\x00" * setup.prefix_size) + int(tf_by_token_doc[token, doc_id].item()).to_bytes(
        setup.tf_size,
        "big",
    )
    print(f"[OKVS key] doc={doc_id}, token={token}, key_prefix={short_hex(okvs_key)}")
    print(f"[Ciphertext check] AES label ciphertext_prefix={short_hex(ciphertext)}")
    print(f"[Decrypt/audit] AES label plaintext_prefix={short_hex(payload)}, tf={int.from_bytes(payload[setup.prefix_size:], 'big')}")
    print(f"[Plain expected] expected_payload_prefix={short_hex(expected_payload)}")
    assert ciphertext != payload
    assert payload == expected_payload

    recovered_tf = client.recover_term_frequencies(response, setup)
    print("[Protocol output] recovered TF:")
    print(recovered_tf)
    assert_close("Protocol 4 recovered TF equals plaintext TF baseline", recovered_tf, expected_tf)


def audit_protocol3_oblivious_filter():
    section("Protocol 3 audit: SimHash projections + OKVS + Paillier/Shamir")

    document_bits = torch.tensor(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.uint8,
    )
    query_bits = document_bits[0].clone()
    print("[Plain input] document SimHash bits:")
    print(document_bits)
    print(f"[Plain input] query bits={query_bits.tolist()}")

    server = Protocol3Server(
        okvs=BinaryOKVS(expansion=2.4),
        he=AdditivePaillier(key_size=64),
        threshold=2,
        projection_count=5,
        simhash_bits=document_bits.shape[1],
        seed=b"protocol234-audit-p3",
    )
    client = Protocol3Client(okvs=server.okvs, shuffle_seed=b"protocol234-audit-p3-client")
    setup = server.build_setup_from_bits(document_bits, chunks=list(range(document_bits.shape[0])))
    plain_candidates = protocol3_plain_projection_candidates(document_bits, query_bits, setup)
    print(
        f"[Public setup] masks={len(setup.masks)}, projection_weight={setup.projection_weight}, "
        f"ciphertext_size={setup.ciphertext_size}"
    )
    print(f"[Plain expected] projected-match candidates={plain_candidates}")
    print(f"[Secret map/audit only] server secret->doc count={len(server._secret_to_document)}")

    first_mask = setup.masks[0]
    doc_key = protocol3_projection_key(tuple(document_bits[0].tolist()), first_mask, mask_id=0)
    query_key = protocol3_projection_key(tuple(query_bits.tolist()), first_mask, mask_id=0)
    encoded_ciphertext = server.okvs.decode(setup.table, query_key)
    decoded_ciphertext_bytes = client.okvs.decode(setup.table, query_key)
    decoded_ciphertexts = client.filter_from_bits(query_bits, setup).shuffled_secret_ciphertexts
    print(f"[Projection 0] mask_indices={list(first_mask)}")
    print(f"[Projection 0] server doc-0 key prefix={short_hex(doc_key)}")
    print(f"[Projection 0] client query key prefix={short_hex(query_key)}")
    print(f"[Projection 0] keys_equal={doc_key == query_key}")
    print(f"[Ciphertext check] encoded OKVS ciphertext bytes prefix={short_hex(encoded_ciphertext)}")
    print(f"[Ciphertext check] client decoded same ciphertext bytes={encoded_ciphertext == decoded_ciphertext_bytes}")

    client_state = client.state
    if client_state is None:
        raise AssertionError("client state was not recorded")
    first_ciphertext = client_state.decoded_ciphertexts[0]
    first_share_plain = server.he.decrypt(first_ciphertext)
    print(f"[Ciphertext check] first Paillier ciphertext={short_hex(first_ciphertext)}")
    print(f"[Decrypt/audit] first Paillier share plaintext={first_share_plain}")
    assert first_ciphertext != first_share_plain

    decrypted_interpolations = [server.he.decrypt(ciphertext) for ciphertext in decoded_ciphertexts]
    known_secret_to_doc = {secret: document.index for secret, document in server._secret_to_document.items()}
    hits = [(secret, known_secret_to_doc[secret]) for secret in decrypted_interpolations if secret in known_secret_to_doc]
    print(f"[Ciphertext check] shuffled encrypted interpolation count={len(decoded_ciphertexts)}")
    print(f"[Decrypt/audit] decrypted interpolation sample={decrypted_interpolations[:8]}")
    print(f"[Decrypt/audit] interpolation hits (secret, doc_id) sample={hits[:8]}")

    candidates = server.recover_candidates(Protocol3ClientMessage(tuple(decoded_ciphertexts)))
    candidate_indices = tuple(document.index for document in candidates)
    print(f"[Protocol output] recovered candidate indices={candidate_indices}")
    assert set(plain_candidates).issubset(set(candidate_indices))

    for left, right in itertools.combinations(range(2), 2):
        left_ct = client_state.decoded_ciphertexts[left]
        right_ct = client_state.decoded_ciphertexts[right]
        encrypted_secret = paillier_interpolate_at_zero(
            left_ct,
            setup.projection_points[left],
            right_ct,
            setup.projection_points[right],
            setup.public_key,
        )
        decrypted_secret = server.he.decrypt(encrypted_secret)
        print(
            "[Interpolation audit] "
            f"left_projection={left}, right_projection={right}, encrypted_secret={short_hex(encrypted_secret)}, "
            f"decrypted_secret={decrypted_secret}, doc_hit={known_secret_to_doc.get(decrypted_secret)}"
        )


def audit_protocol2_secret_shared_bm25():
    section("Protocol 2 audit: secret-shared BM25 components vs plaintext BM25")

    config = PiscesConfig(top_k=2)
    tf = torch.tensor(
        [
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    lengths = torch.tensor([9.0, 7.0, 11.0, 8.0])
    average_length = default_average_length(lengths)
    plain = bm25_components_from_tf(
        tf,
        lengths,
        num_documents=tf.shape[0],
        average_document_length=average_length,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    weighted_tf = plain.idf.unsqueeze(0) * (config.bm25_k1 + 1.0) * tf
    denominator = tf + plain.length_norm.unsqueeze(-1)
    contributions = weighted_tf / denominator

    print("[Plain input] TF from Protocol 4 boundary:")
    print(tf)
    print(f"[Plain input] document_lengths={lengths.tolist()}, average_length={average_length:.6f}")
    print(f"[Plain expected] df={plain.df.tolist()}")
    print(f"[Plain expected] idf={plain.idf.tolist()}")
    print("[Plain expected] weighted_tf numerator:")
    print(weighted_tf)
    print("[Plain expected] denominator=tf+length_norm:")
    print(denominator)
    print("[Plain expected] BM25 contributions:")
    print(contributions)
    print(f"[Plain expected] BM25 scores={plain.scores.tolist()}")

    weighted_shares = share_data(weighted_tf)
    tf_shares = share_data(tf)
    length_norm_shares = share_data(plain.length_norm)
    weighted0, weighted1 = weighted_shares[0][0], weighted_shares[1][0]
    tf0, tf1 = tf_shares[0][0], tf_shares[1][0]
    length0, length1 = length_norm_shares[0][0], length_norm_shares[1][0]

    print("[Ciphertext/share check] weighted_tf share0 ring sample:")
    print(weighted0.item.tensor.reshape(-1)[:6])
    print("[Ciphertext/share check] weighted_tf share1 ring sample:")
    print(weighted1.item.tensor.reshape(-1)[:6])
    print("[Ciphertext/share check] one share alone does not equal plaintext weighted_tf sample:")
    print(weighted0.item.convert_to_real_field().reshape(-1)[:6])

    restored_weighted = restore(weighted0, weighted1)
    restored_tf = restore(tf0, tf1)
    restored_length = restore(length0, length1)
    restored_denominator = restored_tf + restored_length.unsqueeze(-1)
    restored_contributions = restored_weighted / restored_denominator
    restored_scores = restored_contributions.sum(dim=-1)

    print("[Decrypt/restore audit] restored weighted_tf:")
    print(restored_weighted)
    print("[Decrypt/restore audit] restored TF:")
    print(restored_tf)
    print("[Decrypt/restore audit] restored length_norm:")
    print(restored_length)
    print("[Decrypt/restore audit] restored denominator:")
    print(restored_denominator)
    print("[Decrypt/restore audit] restored contributions:")
    print(restored_contributions)
    print(f"[Decrypt/restore audit] restored BM25 scores={restored_scores.tolist()}")

    assert_close("Protocol 2 restored weighted_tf equals plaintext numerator", restored_weighted, weighted_tf)
    assert_close("Protocol 2 restored TF equals plaintext TF", restored_tf, tf)
    assert_close("Protocol 2 restored length_norm equals plaintext server length term", restored_length, plain.length_norm)
    assert_close("Protocol 2 restored denominator equals plaintext denominator", restored_denominator, denominator)
    assert_close("Protocol 2 restored contributions equal plaintext BM25 contributions", restored_contributions, plain.contributions)
    assert_close("Protocol 2 restored scores equal plaintext BM25 scores", restored_scores, plain.scores)


if __name__ == "__main__":
    audit_protocol4_dense_labeled_psi()
    audit_protocol3_oblivious_filter()
    audit_protocol2_secret_shared_bm25()
    print("\npisces protocol2/3/4 ciphertext audit ok")
