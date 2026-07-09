import os
import random
import resource
import sys
import time
from collections import defaultdict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces import PiscesConfig
from NssMPC.application.rag.pisces.protocol1 import Protocol1Client, Protocol1Server
from NssMPC.application.rag.pisces.protocol3 import AdditivePaillier, Protocol3Client, Protocol3Server
from NssMPC.application.rag.pisces.protocol4 import Protocol4Client, Protocol4Server
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


PAPER_PROTOCOL4 = {
    # Pisces Table 5, Multi-instance Labeled PSI.
    "squad_dev_q8": {"time_s": 0.008, "upload_mb": 0.0004, "download_mb": 1.49},
    "squad_train_q8": {"time_s": 0.099, "upload_mb": 0.0004, "download_mb": 22.64},
    "hotpot_dev_distractor_q8": {"time_s": 2.35, "upload_mb": 0.0006, "download_mb": 79.82},
    "hotpot_dev_fullwiki_q8": {"time_s": 2.59, "upload_mb": 0.0006, "download_mb": 81.44},
}


PAPER_SEMANTIC = {
    # Pisces Table 4, Coarse-to-Fine Strategy, Top-10 semantic retrieval.
    "squad_dev": {"time_s": 3.41, "upload_mb": 21.96, "download_mb": 13.93, "accuracy_pct": 75.96},
    "squad_train": {"time_s": 4.46, "upload_mb": 43.51, "download_mb": 118.29, "accuracy_pct": 74.86},
    "hotpot_dev_distractor": {"time_s": 20.10, "upload_mb": 324.90, "download_mb": 1439.70, "accuracy_pct": 79.80},
    "hotpot_dev_fullwiki": {"time_s": 20.76, "upload_mb": 330.69, "download_mb": 1467.45, "accuracy_pct": 78.23},
}


PROTOCOL4_PROFILES = {
    "quick": {
        "paper_key": None,
        "num_docs": 100,
        "vocab_size": 1000,
        "terms_per_doc": 5,
        "query_terms": 8,
    },
    "squad_dev_q8": {
        "paper_key": "squad_dev_q8",
        "num_docs": 1204,
        "vocab_size": 1000,
        "terms_per_doc": 25,
        "query_terms": 8,
    },
    "squad_train_q8": {
        "paper_key": "squad_train_q8",
        "num_docs": 19029,
        "vocab_size": 1000,
        "terms_per_doc": 25,
        "query_terms": 8,
    },
    "hotpot_dev_distractor_q8": {
        "paper_key": "hotpot_dev_distractor_q8",
        "num_docs": 269602,
        "vocab_size": 1000,
        "terms_per_doc": 6,
        "query_terms": 8,
    },
}


def env_int(name, default):
    return int(os.environ.get(name, default))


def mib_from_maxrss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def mb(num_bytes):
    return num_bytes / (1024 * 1024)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_ratio(local, paper):
    if paper in (None, 0):
        return "n/a"
    return f"{local / paper:.2f}x"


def make_sparse_tf(*, num_docs, vocab_size, terms_per_doc, seed):
    rng = random.Random(seed)
    entries = []
    expected_by_token = defaultdict(dict)
    for doc_id in range(num_docs):
        terms = rng.sample(range(vocab_size), terms_per_doc)
        for token in terms:
            tf = rng.randint(1, 4)
            entries.append((token, doc_id, tf))
            expected_by_token[token][doc_id] = tf
    return entries, expected_by_token


def protocol4_benchmark(profile_name):
    profile = PROTOCOL4_PROFILES[profile_name]
    seed = env_int("PISCES_BENCH_SEED", 20260624)
    expansion = float(os.environ.get("PISCES_BENCH_OKVS_EXPANSION", "2.0"))

    section(f"Protocol 4 benchmark: {profile_name}")
    print(
        "[Input] "
        f"num_docs={profile['num_docs']}, vocab_size={profile['vocab_size']}, "
        f"terms_per_doc={profile['terms_per_doc']}, query_terms={profile['query_terms']}, "
        f"okvs_expansion={expansion}"
    )

    t0 = time.perf_counter()
    entries, expected_by_token = make_sparse_tf(
        num_docs=profile["num_docs"],
        vocab_size=profile["vocab_size"],
        terms_per_doc=profile["terms_per_doc"],
        seed=seed,
    )
    generation_s = time.perf_counter() - t0
    active_tokens = sorted(expected_by_token)
    query = torch.tensor(active_tokens[: profile["query_terms"]], dtype=torch.long)
    print(f"[Generate] time={generation_s:.3f}s, entries={len(entries):,}, maxrss={mib_from_maxrss():.1f} MiB")

    params = DHOPRFParams()
    server = Protocol4Server(
        okvs=BinaryOKVS(expansion=expansion, seed=b"paper-bench-protocol4-okvs"),
        oprf_server=DHOPRFServer(params=params, secret_key=246813579),
    )
    client = Protocol4Client(okvs=server.okvs, oprf_client=DHOPRFClient(params=params))

    t0 = time.perf_counter()
    setup = server.build_setup_from_entries(num_docs=profile["num_docs"], entries=entries)
    setup_s = time.perf_counter() - t0
    setup_bytes = setup.table.size * setup.table.value_size + len(setup.table.seed)

    t0 = time.perf_counter()
    request = client.make_query(query)
    request_s = time.perf_counter() - t0
    request_bytes = len(request.elements) * params.element_size

    t0 = time.perf_counter()
    response = server.evaluate_oprf(request)
    response_s = time.perf_counter() - t0
    response_bytes = len(response.elements) * params.element_size

    t0 = time.perf_counter()
    recovered = client.recover_term_frequencies(response, setup)
    recover_s = time.perf_counter() - t0

    mismatches = []
    for query_pos, token in enumerate(query.tolist()):
        expected_docs = expected_by_token[token]
        actual_nonzero = torch.nonzero(recovered[:, query_pos] > 0, as_tuple=False).reshape(-1).tolist()
        actual_docs = {doc_id: int(recovered[doc_id, query_pos].item()) for doc_id in actual_nonzero}
        if actual_docs != expected_docs:
            mismatches.append((token, len(expected_docs), len(actual_docs)))

    local = {
        "setup_time_s": setup_s,
        "online_time_s": request_s + response_s + recover_s,
        "recover_time_s": recover_s,
        "client_upload_mb": mb(request_bytes),
        "server_download_mb": mb(setup_bytes + response_bytes),
        "setup_download_mb": mb(setup_bytes),
        "request_mb": mb(request_bytes),
        "response_mb": mb(response_bytes),
        "okvs_slots": setup.table.size,
        "okvs_value_size": setup.table.value_size,
        "mismatches": len(mismatches),
    }
    print(
        "[Local] "
        f"setup_time={setup_s:.3f}s, online_time={local['online_time_s']:.3f}s "
        f"(request={request_s:.3f}s, response={response_s:.3f}s, recover={recover_s:.3f}s)"
    )
    print(
        "[Local] "
        f"client_upload={local['client_upload_mb']:.6f} MB, "
        f"server_download={local['server_download_mb']:.2f} MB, "
        f"slots={setup.table.size:,}, value_size={setup.table.value_size}, mismatches={local['mismatches']}"
    )
    assert not mismatches

    paper_key = profile["paper_key"]
    if paper_key:
        paper = PAPER_PROTOCOL4[paper_key]
        print(
            "[Paper Table 5] "
            f"time={paper['time_s']}s, upload={paper['upload_mb']} MB, download={paper['download_mb']} MB"
        )
        print(
            "[Compare] "
            f"online_time_ratio={format_ratio(local['online_time_s'], paper['time_s'])}, "
            f"upload_ratio={format_ratio(local['client_upload_mb'], paper['upload_mb'])}, "
            f"download_ratio={format_ratio(local['server_download_mb'], paper['download_mb'])}"
        )
    else:
        print("[Paper] quick profile has no direct paper row; use squad_train_q8 or hotpot_dev_distractor_q8 for paper comparison.")
    return local


def protocol1_synthetic_accuracy():
    section("Protocol 1/3 synthetic accuracy sanity")
    num_docs = env_int("PISCES_P1_DOCS", 64)
    dim = env_int("PISCES_P1_DIM", 32)
    top_k = env_int("PISCES_P1_TOPK", 5)
    seed = env_int("PISCES_P1_SEED", 20260624)
    generator = torch.Generator()
    generator.manual_seed(seed)
    documents = torch.randn(num_docs, dim, generator=generator)
    query = documents[0:1].clone()
    payload = torch.arange(num_docs, dtype=torch.float32).unsqueeze(-1)

    config = PiscesConfig(top_k=top_k, simhash_bits=32, hamming_threshold=2)
    p3_server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"paper-bench-protocol1-okvs"),
        he=AdditivePaillier(key_size=64),
        threshold=2,
        projection_count=8,
        seed=b"paper-bench-protocol1",
        simhash_bits=32,
    )
    server = Protocol1Server(config=config, protocol3=p3_server)
    client = Protocol1Client(config=config, protocol3=Protocol3Client(shuffle_seed=b"paper-bench-protocol1-client"))

    t0 = time.perf_counter()
    setup = server.build_filter_setup(documents, chunks=list(range(num_docs)))
    message = client.make_filter_query(query, setup)
    candidates = server.recover_candidates(message, num_docs=num_docs)
    p3_s = time.perf_counter() - t0

    from NssMPC.application.rag.pisces.protocol1 import protocol1_finish_from_candidate_mask

    result = protocol1_finish_from_candidate_mask(
        query,
        documents,
        candidate_mask=candidates.candidate_mask,
        top_k=top_k,
        document_payload_shares=payload,
    )
    plaintext_top = set(torch.topk((query * documents).sum(dim=-1), top_k).indices.tolist())
    protocol_top = set(int(value.item()) for value in result.documents.reshape(-1))
    overlap = len(plaintext_top & protocol_top) / max(1, top_k)
    candidate_count = int(candidates.candidate_mask.sum().item())
    print(
        "[Local synthetic] "
        f"docs={num_docs}, dim={dim}, top_k={top_k}, candidates={candidate_count}, "
        f"topk_overlap_with_plain={overlap * 100:.2f}%, p3_time={p3_s:.3f}s"
    )
    print(
        "[Paper Table 2/4 note] This synthetic test is not dataset-comparable. "
        "Use real ClapNQ/SQuAD/Hotpot embeddings and qrels to compare paper semantic accuracy."
    )
    return {"candidate_count": candidate_count, "topk_overlap": overlap, "p3_time_s": p3_s}


def main():
    profile = os.environ.get("PISCES_BENCH_PROFILE", "quick")
    if profile not in PROTOCOL4_PROFILES:
        raise ValueError(f"unknown PISCES_BENCH_PROFILE={profile}; choices={sorted(PROTOCOL4_PROFILES)}")
    run_p1 = os.environ.get("PISCES_BENCH_P1", "1") == "1"
    protocol4_benchmark(profile)
    if run_p1:
        protocol1_synthetic_accuracy()
    print("pisces paper benchmark report ok")


if __name__ == "__main__":
    main()
