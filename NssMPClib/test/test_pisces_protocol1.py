import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig
from NssMPC.application.rag.pisces.protocol1 import (
    Protocol1Client,
    Protocol1Server,
    protocol1_finish_from_candidate_mask,
)
from NssMPC.application.rag.pisces.protocol3 import AdditivePaillier, Protocol3Client, Protocol3Server
from NssMPC.crypto.primitives.okvs import BinaryOKVS


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_protocol1_semantic_retrieval_pipeline():
    section("1. Protocol 1 semantic retrieval pipeline")
    query = torch.tensor([[1.0, 0.5, -0.25, 0.75]])
    documents = torch.tensor(
        [
            [1.0, 0.5, -0.25, 0.75],
            [0.9, 0.4, -0.1, 0.6],
            [-1.0, -0.5, 0.25, -0.75],
            [0.2, -0.8, 0.3, -0.1],
        ]
    )
    payload = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    config = PiscesConfig(top_k=1, simhash_bits=16, hamming_threshold=1)
    p3_server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"protocol1-okvs"),
        he=AdditivePaillier(key_size=64),
        threshold=1,
        projection_count=6,
        seed=b"protocol1-test",
        simhash_bits=16,
    )
    server = Protocol1Server(config=config, protocol3=p3_server)
    client = Protocol1Client(config=config, protocol3=Protocol3Client(shuffle_seed=b"protocol1-client"))

    print(f"[Input] documents shape={tuple(documents.shape)}, payload shape={tuple(payload.shape)}")
    print("[Expected] Protocol 3 finds semantic candidates, then Protocol 1 scores only candidates before Top-K/PIR-to-share.")
    setup = server.build_filter_setup(documents, chunks=list(range(documents.shape[0])))
    message = client.make_filter_query(query, setup)
    candidates = server.recover_candidates(message, num_docs=documents.shape[0])
    print(
        f"[P3] candidates={candidates.candidate_indices}, mask={candidates.candidate_mask.tolist()}, "
        f"bucket_capacity={setup.bucket_capacity}"
    )

    result = protocol1_finish_from_candidate_mask(
        query,
        documents,
        candidate_mask=candidates.candidate_mask,
        top_k=config.top_k,
        document_payload_shares=payload,
    )
    expected_scores = (query * documents).sum(dim=-1)
    expected_masked = expected_scores + (1.0 - candidates.candidate_mask) * -1000000.0
    expected_index = int(torch.topk(expected_masked, config.top_k).indices[0].item())
    print(f"[Fine] expected_scores={expected_scores.tolist()}")
    print(f"[Fine] masked_scores={result.scores.tolist()}")
    print("[Top-K] indicators:")
    print(result.indicators)
    print(f"[PIR-to-share] audit={result.pir.audit}")
    print("[PIR-to-share] selected payload:")
    print(result.documents)

    assert torch.equal(result.scores, expected_masked)
    assert torch.equal(result.documents, payload[[expected_index]])
    print("[Check] Protocol 1 pipeline matches the plain masked top-k baseline")


if __name__ == "__main__":
    test_protocol1_semantic_retrieval_pipeline()
    print("pisces protocol1 tests ok")
