import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig, PiscesRetriever
from NssMPC.application.rag.pisces.ops import bm25_components_from_tf, default_average_length, simhash
from NssMPC.application.rag.pisces.pir import suda_pir_to_share
from NssMPC.application.rag.pisces.protocol1 import (
    Protocol1Client,
    Protocol1Server,
    protocol1_finish_from_candidate_mask,
)
from NssMPC.application.rag.pisces.protocol3 import (
    AdditivePaillier,
    Protocol3Client,
    Protocol3Server,
    protocol3_plain_projection_candidates,
)
from NssMPC.application.rag.pisces.protocol4 import Protocol4Client, Protocol4Server
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


def section(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def recall_at_k(predicted_indices, relevant_indices):
    predicted = set(int(index) for index in predicted_indices)
    relevant = set(int(index) for index in relevant_indices)
    if not relevant:
        return 1.0
    return len(predicted & relevant) / len(relevant)


def build_demo_inputs():
    query_embedding = torch.tensor([[0.9, -0.4, 0.7, 0.2, -0.6, 0.3, 0.8, -0.1]], dtype=torch.float32)
    document_embeddings = torch.tensor(
        [
            [0.9, -0.4, 0.7, 0.2, -0.6, 0.3, 0.8, -0.1],
            [0.72, -0.32, 0.56, 0.16, -0.48, 0.24, 0.64, -0.08],
            [-0.8, 0.5, -0.7, -0.1, 0.7, -0.2, -0.9, 0.2],
            [0.1, 0.9, 0.2, -0.5, -0.1, 0.4, -0.3, 0.7],
            [0.4, -0.1, 0.3, 0.8, -0.2, 0.1, 0.5, 0.6],
            [-0.2, -0.7, 0.4, 0.3, 0.6, 0.5, -0.1, -0.4],
        ],
        dtype=torch.float32,
    )
    document_payload = torch.tensor(
        [
            [101, 2001, 2002, 102, 0, 0],
            [101, 2011, 2012, 102, 0, 0],
            [101, 2021, 2022, 102, 0, 0],
            [101, 2031, 2032, 102, 0, 0],
            [101, 2041, 2042, 102, 0, 0],
            [101, 2051, 2052, 102, 0, 0],
        ],
        dtype=torch.float32,
    )

    tf_by_token_doc = torch.zeros(12, document_embeddings.shape[0], dtype=torch.float32)
    tf_by_token_doc[2] = torch.tensor([3, 2, 0, 0, 1, 0], dtype=torch.float32)
    tf_by_token_doc[5] = torch.tensor([0, 1, 4, 0, 0, 2], dtype=torch.float32)
    tf_by_token_doc[7] = torch.tensor([1, 0, 0, 5, 2, 0], dtype=torch.float32)
    tf_by_token_doc[9] = torch.tensor([0, 0, 1, 1, 0, 3], dtype=torch.float32)
    document_lengths = tf_by_token_doc.sum(dim=0).clamp_min(1.0)
    query_tokens = torch.tensor([2, 5, 7], dtype=torch.long)

    return query_embedding, document_embeddings, document_payload, tf_by_token_doc, document_lengths, query_tokens


def run_semantic_path(config, query_embedding, document_embeddings, document_payload):
    section("1. Protocol 1 semantic path: Protocol 3 filter -> fine score -> top-k -> PIR-to-share")

    protocol3_server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"pisces-e2e-p3-okvs"),
        he=AdditivePaillier(key_size=64),
        threshold=2,
        projection_count=8,
        seed=b"pisces-e2e-p3",
        simhash_bits=config.simhash_bits,
    )
    server = Protocol1Server(config=config, protocol3=protocol3_server)
    client = Protocol1Client(config=config, protocol3=Protocol3Client(shuffle_seed=b"pisces-e2e-p3-client"))

    setup = server.build_filter_setup(document_embeddings, chunks=list(range(document_embeddings.shape[0])))
    message = client.make_filter_query(query_embedding, setup)
    candidates = server.recover_candidates(message, num_docs=document_embeddings.shape[0])
    document_bits = simhash(document_embeddings, bits=setup.simhash_bits)
    query_bits = simhash(query_embedding, bits=setup.simhash_bits).reshape(-1)
    plain_projection_candidates = protocol3_plain_projection_candidates(document_bits, query_bits, setup)
    print(
        "[P3 setup] "
        f"simhash_bits={setup.simhash_bits}, masks={len(setup.masks)}, "
        f"projection_weight={setup.projection_weight}, bucket_capacity={setup.bucket_capacity}, "
        f"okvs_slots={setup.table.size}"
    )
    print(f"[P3 plain baseline] projection candidates={plain_projection_candidates}")
    print(f"[P3 output] candidate_indices={candidates.candidate_indices}")
    print(f"[P3 output] candidate_mask={candidates.candidate_mask.tolist()}")
    assert set(candidates.candidate_indices) == set(plain_projection_candidates)

    result = protocol1_finish_from_candidate_mask(
        query_embedding,
        document_embeddings,
        candidate_mask=candidates.candidate_mask,
        top_k=config.top_k,
        document_payload_shares=document_payload,
    )
    plain_scores = (query_embedding * document_embeddings).sum(dim=-1)
    plain_masked = plain_scores + (1.0 - candidates.candidate_mask) * -1000000.0
    expected_indices = torch.topk(plain_masked, config.top_k).indices
    expected_payload = document_payload[expected_indices]

    print(f"[Fine score] plain_scores={plain_scores.tolist()}")
    print(f"[Fine score] masked_scores={result.scores.tolist()}")
    print("[Top-k] semantic indicators:")
    print(result.indicators)
    print(f"[PIR-to-share] semantic audit={result.pir.audit}")
    print("[Protocol output] semantic retrieved payload:")
    print(result.documents)
    print(f"[Plain expected] semantic top-k indices={expected_indices.tolist()}")

    assert torch.equal(result.scores, plain_masked)
    assert torch.equal(result.documents, expected_payload)
    return result, expected_indices


def run_lexical_path(config, document_payload, tf_by_token_doc, document_lengths, query_tokens):
    section("2. Protocol 2 lexical path: Protocol 4 TF recovery -> BM25 -> top-k -> PIR-to-share")

    params = DHOPRFParams()
    okvs = BinaryOKVS(expansion=2.4, seed=b"pisces-e2e-p4-okvs")
    server = Protocol4Server(okvs=okvs, oprf_server=DHOPRFServer(params=params, secret_key=24681357))
    client = Protocol4Client(okvs=okvs, oprf_client=DHOPRFClient(params=params))

    setup = server.build_setup(tf_by_token_doc)
    request = client.make_query(query_tokens)
    response = server.evaluate_oprf(request)
    recovered_tf = client.recover_term_frequencies(response, setup)
    expected_tf = tf_by_token_doc[query_tokens].T.contiguous()

    print(
        "[P4 setup] "
        f"num_docs={setup.num_docs}, okvs_slots={setup.table.size}, value_size={setup.table.value_size}, "
        f"query_tokens={query_tokens.tolist()}"
    )
    print(f"[P4 message] blinded_request_count={len(request.elements)}, response_count={len(response.elements)}")
    print("[P4 output] recovered TF [docs, query_terms]:")
    print(recovered_tf)
    print("[Plain expected] TF [docs, query_terms]:")
    print(expected_tf)
    assert torch.equal(recovered_tf, expected_tf)

    bm25 = bm25_components_from_tf(
        recovered_tf,
        document_lengths,
        num_documents=recovered_tf.shape[0],
        average_document_length=default_average_length(document_lengths),
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    retriever = PiscesRetriever(config)
    scores, indicators, _ = retriever.lexical_path_from_scores(bm25.scores)
    pir = suda_pir_to_share(indicators, document_payload)
    expected_indices = torch.topk(bm25.scores, config.top_k).indices
    expected_payload = document_payload[expected_indices]

    print(f"[P2 BM25] document_lengths={document_lengths.tolist()}")
    print(f"[P2 BM25] df={bm25.df.tolist()}")
    print(f"[P2 BM25] idf={bm25.idf.tolist()}")
    print("[P2 BM25] contributions:")
    print(bm25.contributions)
    print(f"[P2 BM25] scores={scores.tolist()}")
    print("[Top-k] lexical indicators:")
    print(indicators)
    print(f"[PIR-to-share] lexical audit={pir.audit}")
    print("[Protocol output] lexical retrieved payload:")
    print(pir.records)
    print(f"[Plain expected] lexical top-k indices={expected_indices.tolist()}")

    assert torch.equal(scores, bm25.scores)
    assert torch.equal(pir.records, expected_payload)
    return scores, indicators, pir, expected_indices


def main():
    config = PiscesConfig(top_k=2, simhash_bits=16, hamming_threshold=2)
    (
        query_embedding,
        document_embeddings,
        document_payload,
        tf_by_token_doc,
        document_lengths,
        query_tokens,
    ) = build_demo_inputs()

    section("Pisces end-to-end functional demo")
    print(f"[Input] query_embedding shape={tuple(query_embedding.shape)}")
    print(f"[Input] document_embeddings shape={tuple(document_embeddings.shape)}")
    print(f"[Input] document_payload shape={tuple(document_payload.shape)}")
    print(f"[Input] tf_by_token_doc shape={tuple(tf_by_token_doc.shape)}")
    print(f"[Input] top_k={config.top_k}")

    semantic_result, semantic_indices = run_semantic_path(
        config,
        query_embedding,
        document_embeddings,
        document_payload,
    )
    _, _, lexical_pir, lexical_indices = run_lexical_path(
        config,
        document_payload,
        tf_by_token_doc,
        document_lengths,
        query_tokens,
    )

    section("3. Dual-path retrieval output")
    semantic_relevant = {0, 1}
    lexical_relevant = {1, 4}
    semantic_recall = recall_at_k(semantic_indices.tolist(), semantic_relevant)
    lexical_recall = recall_at_k(lexical_indices.tolist(), lexical_relevant)
    fused_context = torch.cat([semantic_result.documents, lexical_pir.records], dim=0)
    print(f"[Semantic] top-k indices={semantic_indices.tolist()}")
    print(f"[Semantic] controlled relevant docs={sorted(semantic_relevant)}, recall@{config.top_k}={semantic_recall:.3f}")
    print(f"[Lexical] top-k indices={lexical_indices.tolist()}")
    print(f"[Lexical] controlled relevant docs={sorted(lexical_relevant)}, recall@{config.top_k}={lexical_recall:.3f}")
    print("[Fused output] semantic payloads followed by lexical payloads:")
    print(fused_context)
    assert semantic_recall == 1.0
    assert lexical_recall == 1.0
    print("\npisces end-to-end functional demo ok")


if __name__ == "__main__":
    main()
