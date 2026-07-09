import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig, PiscesRetriever
from NssMPC.application.rag.pisces.ops import bm25_components_from_tf, default_average_length
from NssMPC.application.rag.pisces.pir import suda_pir_to_share


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_protocol2_plain_bm25_topk_pir_pipeline():
    section("1. Protocol 2 BM25 -> Top-K -> PIR-to-share audit")
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
    payload = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    config = PiscesConfig(top_k=2)
    retriever = PiscesRetriever(config)

    print("[Input] TF recovered from Protocol 4 boundary [num_docs, query_terms]:")
    print(tf)
    print(f"[Input] document lengths={lengths.tolist()}, payload shape={tuple(payload.shape)}")
    bm25 = bm25_components_from_tf(
        tf,
        lengths,
        num_documents=tf.shape[0],
        average_document_length=default_average_length(lengths),
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    print(f"[Protocol 2] df={bm25.df.tolist()}")
    print(f"[Protocol 2] idf={bm25.idf.tolist()}")
    print(f"[Protocol 2] length_norm={bm25.length_norm.tolist()}")
    print("[Protocol 2] contributions:")
    print(bm25.contributions)
    print(f"[Protocol 2] scores={bm25.scores.tolist()}")

    scores, indicators, _ = retriever.lexical_path_from_scores(bm25.scores)
    pir = suda_pir_to_share(indicators, payload)
    expected_indices = torch.topk(bm25.scores, config.top_k).indices
    expected_docs = payload[expected_indices]
    print("[Top-K] indicators:")
    print(indicators)
    print(f"[Top-K] expected indices={expected_indices.tolist()}")
    print(f"[PIR-to-share] audit={pir.audit}")
    print("[PIR-to-share] selected payload:")
    print(pir.records)

    assert torch.equal(scores, bm25.scores)
    assert torch.equal(pir.records, expected_docs)
    print("[Check] Protocol 2 plain audit pipeline selects the same payload as torch.topk baseline")


if __name__ == "__main__":
    test_protocol2_plain_bm25_topk_pir_pipeline()
    print("pisces protocol2 tests ok")
