import os
import random
import resource
import sys
import time
from collections import defaultdict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces.protocol4 import Protocol4Client, Protocol4Server
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


def env_int(name, default):
    return int(os.environ.get(name, default))


def mib_from_maxrss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


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


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    num_docs = env_int("PISCES_SCALE_DOCS", 5000)
    vocab_size = env_int("PISCES_SCALE_VOCAB", 8000)
    terms_per_doc = env_int("PISCES_SCALE_TERMS_PER_DOC", 20)
    query_terms = env_int("PISCES_SCALE_QUERY_TERMS", 8)
    seed = env_int("PISCES_SCALE_SEED", 20260615)
    expansion = float(os.environ.get("PISCES_SCALE_OKVS_EXPANSION", "2.0"))

    section("Pisces Protocol 4 synthetic SQuAD-scale-ish sparse benchmark")
    print(
        "[Scale] "
        f"num_docs={num_docs}, vocab_size={vocab_size}, terms_per_doc={terms_per_doc}, "
        f"nonzero_entries={num_docs * terms_per_doc}, query_terms={query_terms}, okvs_expansion={expansion}"
    )
    print("[Note] This uses sparse (token, doc, tf) entries, not a dense [vocab, docs] matrix.")

    t0 = time.perf_counter()
    entries, expected_by_token = make_sparse_tf(
        num_docs=num_docs,
        vocab_size=vocab_size,
        terms_per_doc=terms_per_doc,
        seed=seed,
    )
    generation_time = time.perf_counter() - t0
    active_tokens = sorted(expected_by_token)
    query = torch.tensor(active_tokens[:query_terms], dtype=torch.long)
    print(
        f"[Generate] time={generation_time:.3f}s, active_tokens={len(active_tokens)}, "
        f"query_tokens={query.tolist()}, maxrss={mib_from_maxrss():.1f} MiB"
    )

    params = DHOPRFParams()
    server = Protocol4Server(
        okvs=BinaryOKVS(expansion=expansion, seed=b"pisces-scale-okvs"),
        oprf_server=DHOPRFServer(params=params, secret_key=246813579),
    )
    client = Protocol4Client(okvs=server.okvs, oprf_client=DHOPRFClient(params=params))

    t0 = time.perf_counter()
    setup = server.build_setup_from_entries(num_docs=num_docs, entries=entries)
    setup_time = time.perf_counter() - t0
    setup_raw_bytes = setup.table.size * setup.table.value_size + len(setup.table.seed)
    print(
        f"[Setup] time={setup_time:.3f}s, okvs_slots={setup.table.size}, "
        f"value_size={setup.table.value_size}, method={setup.table.method}, "
        f"raw_setup_bytes={setup_raw_bytes:,}, maxrss={mib_from_maxrss():.1f} MiB"
    )

    t0 = time.perf_counter()
    request = client.make_query(query)
    request_time = time.perf_counter() - t0
    request_raw_bytes = len(request.elements) * params.element_size
    print(
        f"[Client request] time={request_time:.3f}s, elements={len(request.elements)}, "
        f"raw_bytes={request_raw_bytes:,}, element_size={params.element_size}"
    )

    t0 = time.perf_counter()
    response = server.evaluate_oprf(request)
    response_time = time.perf_counter() - t0
    response_raw_bytes = len(response.elements) * params.element_size
    print(f"[Server response] time={response_time:.3f}s, elements={len(response.elements)}, raw_bytes={response_raw_bytes:,}")

    t0 = time.perf_counter()
    recovered = client.recover_term_frequencies(response, setup)
    recover_time = time.perf_counter() - t0
    print(f"[Recover] time={recover_time:.3f}s, recovered_shape={tuple(recovered.shape)}, maxrss={mib_from_maxrss():.1f} MiB")

    mismatches = []
    for query_pos, token in enumerate(query.tolist()):
        expected_docs = expected_by_token[token]
        actual_nonzero = torch.nonzero(recovered[:, query_pos] > 0, as_tuple=False).reshape(-1).tolist()
        actual_docs = {doc_id: int(recovered[doc_id, query_pos].item()) for doc_id in actual_nonzero}
        if actual_docs != expected_docs:
            mismatches.append((token, len(expected_docs), len(actual_docs)))
    print(f"[Check] query_terms_checked={len(query)}, mismatches={mismatches[:5]}")
    assert not mismatches

    total_raw_comm = setup_raw_bytes + request_raw_bytes + response_raw_bytes
    print(
        f"[Communication estimate] setup={setup_raw_bytes:,} bytes, "
        f"request={request_raw_bytes:,} bytes, response={response_raw_bytes:,} bytes, "
        f"total={total_raw_comm:,} bytes"
    )
    print("pisces protocol4 synthetic scale test ok")


if __name__ == "__main__":
    main()
