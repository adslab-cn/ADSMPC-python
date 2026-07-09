import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
PISCES_PLAINTEXT_ROOT = Path("/home/adslab/pazika/pisces")
sys.path.append(str(PISCES_PLAINTEXT_ROOT))
sys.path.append(str(PISCES_PLAINTEXT_ROOT / "scripts"))

from verify_datasets import prepare_hotpotqa, prepare_squad  # noqa: E402
from pisces_plaintext.bm25 import BM25Scorer  # noqa: E402

from NssMPC.application.rag.pisces import PiscesConfig, PiscesRetriever  # noqa: E402
from NssMPC.application.rag.pisces.ops import bm25_components_from_tf, default_average_length  # noqa: E402
from NssMPC.application.rag.pisces.pir import suda_pir_to_share  # noqa: E402
from NssMPC.application.rag.pisces.protocol1 import (  # noqa: E402
    Protocol1Client,
    Protocol1Server,
    protocol1_finish_from_candidate_mask,
)
from NssMPC.application.rag.pisces.protocol3 import AdditivePaillier, Protocol3Client, Protocol3Server  # noqa: E402
from NssMPC.application.rag.pisces.protocol4 import Protocol4Client, Protocol4Server  # noqa: E402
from NssMPC.crypto.primitives.okvs import BinaryOKVS  # noqa: E402
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer  # noqa: E402


DEFAULT_CACHE = PISCES_PLAINTEXT_ROOT / ".embedding-cache"
DEFAULT_SETUP_CACHE = Path("data/pisces_setup_cache")
DATASETS = {
    "hotpotqa_dev_distractor": {
        "path": PISCES_PLAINTEXT_ROOT / "hotpot/hotpot_dev_distractor_v1.json",
        "prepare": prepare_hotpotqa,
        "chunks": "hotpotqa_dev_distractor_chunks_ibm-granite_granite-embedding-small-english-r2_cls_max8192_269602.npy",
        "queries": "hotpotqa_dev_distractor_queries_7405_ibm-granite_granite-embedding-small-english-r2_cls_max8192_7405.npy",
        "terms": "hotpotqa_dev_distractor_bert_terms_bert-base-uncased_269602.pkl",
    },
    "squad_dev_v2": {
        "path": Path("data/squad/dev-v2.0.json"),
        "prepare": prepare_squad,
        "chunks": "squad_dev_v2_chunks_ibm-granite_granite-embedding-small-english-r2_cls_max8192_1204.npy",
        "queries": "squad_dev_v2_queries_5928_ibm-granite_granite-embedding-small-english-r2_cls_max8192_5928.npy",
        "terms": "squad_dev_v2_bert_terms_bert-base-uncased_1204.pkl",
    },
    "squad_train_v2": {
        "path": Path("data/squad/train-v2.0.json"),
        "prepare": prepare_squad,
        "chunks": "squad_train_v2_chunks_ibm-granite_granite-embedding-small-english-r2_cls_max8192_19029.npy",
        "queries": "squad_train_v2_queries_1000_ibm-granite_granite-embedding-small-english-r2_cls_max8192_1000.npy",
        "terms": "squad_train_v2_bert_terms_bert-base-uncased_19029.pkl",
    },
}


def section(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current Pisces pipeline on cached HotpotQA data.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="hotpotqa_dev_distractor")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--corpus-limit", type=int, default=2000)
    parser.add_argument("--limit-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--protocol-docs", type=int, default=1000)
    parser.add_argument("--semantic-candidates", type=int, default=32)
    parser.add_argument("--simhash-bits", type=int, default=16)
    parser.add_argument("--p3-projections", type=int, default=8)
    parser.add_argument("--p3-threshold", type=int, default=2)
    parser.add_argument("--he-key-size", type=int, default=64)
    parser.add_argument("--verbose-queries", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--setup-mode", choices=["offline", "per-query"], default="offline")
    parser.add_argument("--setup-cache-dir", default=str(DEFAULT_SETUP_CACHE))
    parser.add_argument("--no-setup-cache", action="store_true")
    parser.add_argument("--report-jsonl", default=None)
    parser.add_argument("--resume-report", action="store_true")
    return parser.parse_args()


class QueryTokenizer:
    def __init__(self):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.tokenizer.model_max_length = 10**12
            self.mode = "bert-base-uncased-local"
        except Exception as exc:
            print(f"[Tokenizer] local bert-base-uncased unavailable, falling back to simple tokenizer: {exc}")
            self.tokenizer = None
            self.mode = "simple-regex"
            self.pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

    def tokenize(self, text):
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        return self.pattern.findall(text.lower())


def load_inputs(args):
    spec = DATASETS[args.dataset]
    dataset_path = Path(args.dataset_path) if args.dataset_path else spec["path"]
    cache_dir = Path(args.cache_dir)
    prepared = spec["prepare"](args.dataset, dataset_path)
    chunk_embeddings = np.load(cache_dir / spec["chunks"], mmap_mode="r")
    query_embeddings = np.load(cache_dir / spec["queries"], mmap_mode="r")
    with (cache_dir / spec["terms"]).open("rb") as handle:
        term_frequencies = pickle.load(handle)
    if len(prepared.chunks) != chunk_embeddings.shape[0] or len(prepared.chunks) != len(term_frequencies):
        raise ValueError(f"{args.dataset} chunk metadata and caches are not aligned")
    if len(prepared.queries) < query_embeddings.shape[0]:
        raise ValueError(f"{args.dataset} has fewer queries than the embedding cache")
    return prepared, chunk_embeddings, query_embeddings, term_frequencies


def choose_queries(prepared, query_embedding_count, corpus_limit, limit_queries):
    selected = []
    for query_index, (query, gold_ids) in enumerate(prepared.queries[:query_embedding_count]):
        if gold_ids and all(gold_id < corpus_limit for gold_id in gold_ids):
            selected.append((query_index, query, set(gold_ids)))
        if len(selected) >= limit_queries:
            break
    if not selected:
        raise ValueError(f"no Hotpot queries have all gold chunks within corpus_limit={corpus_limit}")
    return selected


def build_bm25_vocab(term_frequencies):
    bm25 = BM25Scorer(term_frequencies)
    vocabulary = {term: index for index, term in enumerate(bm25.stats.document_frequency)}
    return bm25, vocabulary


def build_global_tf_entries(term_frequencies, vocabulary):
    entries = []
    for doc_id, counter in enumerate(term_frequencies):
        for term, frequency in counter.items():
            token_id = vocabulary.get(term)
            if token_id is not None and frequency > 0:
                entries.append((token_id, doc_id, int(frequency)))
    return entries


def setup_cache_path(args, vocabulary, term_frequencies):
    term_spec = DATASETS[args.dataset]["terms"]
    term_path = Path(args.cache_dir) / term_spec
    stat = term_path.stat()
    fingerprint = {
        "version": 1,
        "dataset": args.dataset,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "term_file": term_spec,
        "term_file_size": stat.st_size,
        "term_file_mtime_ns": stat.st_mtime_ns,
        "corpus_limit": args.corpus_limit,
        "protocol_docs": min(args.protocol_docs, args.corpus_limit),
        "vocab_size": len(vocabulary),
        "doc_count": len(term_frequencies),
        "top_k": args.top_k,
        "simhash_bits": args.simhash_bits,
        "p3_projections": args.p3_projections,
        "p3_threshold": args.p3_threshold,
        "he_key_size": args.he_key_size,
        "bm25_k1": PiscesConfig().bm25_k1,
        "bm25_b": PiscesConfig().bm25_b,
    }
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return Path(args.setup_cache_dir) / f"{args.dataset}_offline_{digest}.pkl", fingerprint


def load_offline_index_from_cache(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    index = payload["index"]
    if not hasattr(index, "p3_setup_wire_bytes"):
        index.p3_setup_wire_bytes = protocol3_setup_wire_bytes(index.p3_setup)
    if not hasattr(index, "p4_setup_wire_bytes"):
        index.p4_setup_wire_bytes = protocol4_setup_wire_bytes(index.p4_setup)
    return index


def save_offline_index_to_cache(path, index, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump({"fingerprint": fingerprint, "index": index}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def int_wire_size(value):
    return max(1, (int(value).bit_length() + 7) // 8)


def okvs_wire_size(table):
    return table.size * table.value_size


def protocol3_setup_wire_bytes(setup):
    public_key_bytes = sum(int_wire_size(value) for value in setup.public_key)
    masks_bytes = sum(len(mask) * 2 for mask in setup.masks)
    points_bytes = sum(int_wire_size(point) for point in setup.projection_points)
    return public_key_bytes + masks_bytes + points_bytes + okvs_wire_size(setup.table)


def protocol3_query_wire_bytes(message):
    return sum(int_wire_size(ciphertext) for ciphertext in message.shuffled_secret_ciphertexts)


def protocol4_setup_wire_bytes(setup):
    return okvs_wire_size(setup.table) + 8 + 2 + 2


def protocol4_query_wire_bytes(request, response, element_size):
    return len(request.elements) * element_size, len(response.elements) * element_size


def append_jsonl(path, record):
    if path is None:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def completed_query_indices(path):
    if path is None:
        return set()
    report_path = Path(path)
    if not report_path.exists():
        return set()
    completed = set()
    with report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "query" and "query_index" in record:
                completed.add(int(record["query_index"]))
    return completed


class OfflinePiscesIndex:
    def __init__(
        self,
        *,
        doc_ids,
        chunk_embeddings,
        term_frequencies,
        vocabulary,
        config,
        args,
    ):
        self.doc_ids = list(doc_ids)
        self.global_to_local = {doc_id: local for local, doc_id in enumerate(self.doc_ids)}
        self.doc_embeddings = torch.tensor(np.asarray(chunk_embeddings[self.doc_ids]), dtype=torch.float32)
        self.payload = torch.tensor(self.doc_ids, dtype=torch.float32).unsqueeze(-1)
        self.term_frequencies = [term_frequencies[doc_id] for doc_id in self.doc_ids]
        self.lengths = torch.tensor([sum(counter.values()) or 1 for counter in self.term_frequencies], dtype=torch.float32)
        self.vocabulary = vocabulary
        self.config = config
        self.args = args

        started = time.perf_counter()
        self.p3_server = Protocol3Server(
            okvs=BinaryOKVS(expansion=3.0, seed=b"real-dataset-offline-p3-okvs"),
            he=AdditivePaillier(key_size=args.he_key_size),
            threshold=args.p3_threshold,
            projection_count=args.p3_projections,
            seed=b"real-dataset-offline-p3",
            simhash_bits=args.simhash_bits,
        )
        self.p1_server = Protocol1Server(config=config, protocol3=self.p3_server)
        self.p3_setup = self.p1_server.build_filter_setup(self.doc_embeddings, chunks=self.doc_ids)
        self.p3_setup_s = time.perf_counter() - started
        self.p3_setup_wire_bytes = protocol3_setup_wire_bytes(self.p3_setup)

        started = time.perf_counter()
        entries = build_global_tf_entries(self.term_frequencies, vocabulary)
        params = DHOPRFParams()
        self.p4_okvs = BinaryOKVS(expansion=2.4, seed=b"real-dataset-offline-p4-okvs")
        self.p4_server = Protocol4Server(
            okvs=self.p4_okvs,
            oprf_server=DHOPRFServer(params=params, secret_key=97531),
        )
        self.p4_client_params = params
        self.p4_setup = self.p4_server.build_setup_from_entries(num_docs=len(self.doc_ids), entries=entries)
        self.p4_setup_s = time.perf_counter() - started
        self.p4_entries = len(entries)
        self.p4_setup_wire_bytes = protocol4_setup_wire_bytes(self.p4_setup)

    def query(self, query_embedding, query_terms):
        timings = {}
        wire = {
            "p3_client_upload_bytes": 0,
            "p4_client_upload_bytes": 0,
            "p4_server_download_bytes": 0,
        }
        query_tensor = torch.tensor(np.asarray(query_embedding), dtype=torch.float32).unsqueeze(0)
        p1_client = Protocol1Client(
            config=self.config,
            protocol3=Protocol3Client(shuffle_seed=b"real-dataset-offline-p3-client"),
        )
        started = time.perf_counter()
        p3_message = p1_client.make_filter_query(query_tensor, self.p3_setup)
        timings["p3_client_filter_s"] = time.perf_counter() - started
        wire["p3_client_upload_bytes"] = protocol3_query_wire_bytes(p3_message)

        started = time.perf_counter()
        candidates = self.p1_server.recover_candidates(p3_message, num_docs=len(self.doc_ids))
        timings["p3_server_recover_s"] = time.perf_counter() - started

        started = time.perf_counter()
        semantic = protocol1_finish_from_candidate_mask(
            query_tensor,
            self.doc_embeddings,
            candidate_mask=candidates.candidate_mask,
            top_k=self.config.top_k,
            document_payload_shares=self.payload,
        )
        semantic_ids = [int(value.item()) for value in semantic.documents.reshape(-1)]
        timings["semantic_score_topk_s"] = time.perf_counter() - started

        started = time.perf_counter()
        query_token_ids = [self.vocabulary[term] for term in query_terms if term in self.vocabulary]
        timings["query_vocab_lookup_s"] = time.perf_counter() - started
        if not query_token_ids:
            lexical_ids = self.doc_ids[: self.config.top_k]
            bm25_scores = torch.zeros(len(self.doc_ids), dtype=torch.float32)
            timings["p4_oprf_s"] = 0.0
            timings["p4_recover_tf_s"] = 0.0
            timings["bm25_topk_pir_s"] = 0.0
        else:
            query_tokens = torch.tensor(query_token_ids, dtype=torch.long)
            p4_client = Protocol4Client(okvs=self.p4_okvs, oprf_client=DHOPRFClient(params=self.p4_client_params))
            started = time.perf_counter()
            p4_request = p4_client.make_query(query_tokens)
            p4_response = self.p4_server.evaluate_oprf(p4_request)
            timings["p4_oprf_s"] = time.perf_counter() - started
            upload, download = protocol4_query_wire_bytes(
                p4_request,
                p4_response,
                self.p4_client_params.element_size,
            )
            wire["p4_client_upload_bytes"] = upload
            wire["p4_server_download_bytes"] = download

            started = time.perf_counter()
            recovered_tf = p4_client.recover_term_frequencies(p4_response, self.p4_setup)
            timings["p4_recover_tf_s"] = time.perf_counter() - started

            started = time.perf_counter()
            bm25 = bm25_components_from_tf(
                recovered_tf,
                self.lengths,
                num_documents=recovered_tf.shape[0],
                average_document_length=default_average_length(self.lengths),
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
            bm25_scores = bm25.scores
            retriever = PiscesRetriever(self.config)
            _, lexical_indicators, _ = retriever.lexical_path_from_scores(bm25_scores)
            lexical_pir = suda_pir_to_share(lexical_indicators, self.payload)
            lexical_ids = [int(value.item()) for value in lexical_pir.records.reshape(-1)]
            timings["bm25_topk_pir_s"] = time.perf_counter() - started

        fused_ids = []
        for doc_id in semantic_ids + lexical_ids:
            if doc_id not in fused_ids:
                fused_ids.append(doc_id)
        return {
            "candidate_size": int(candidates.candidate_mask.sum().item()),
            "semantic_ids": semantic_ids,
            "lexical_ids": lexical_ids,
            "fused_ids": fused_ids[: self.config.top_k],
            "bm25_scores": bm25_scores,
            "query_token_count": len(query_terms),
            "query_vocab_hit_count": len(query_token_ids),
            "timings": timings,
            "wire": wire,
        }


def lexical_pool(query_terms, bm25, vocabulary, term_frequencies, *, top_k):
    query_counts = Counter(term for term in query_terms if term in vocabulary)
    if not query_counts:
        return [(index, 0.0) for index in range(min(top_k, len(term_frequencies)))]
    scores = []
    for doc_id, counter in enumerate(term_frequencies):
        doc_length = bm25.lengths[doc_id] or 1
        score = 0.0
        for term, query_count in query_counts.items():
            frequency = counter.get(term, 0)
            if frequency:
                score += query_count * bm25._idf(term) * bm25._term_relevance(frequency, doc_length)
        scores.append(score)
    if len(scores) <= top_k:
        ranked = np.argsort(scores)[::-1]
    else:
        raw = np.asarray(scores, dtype=np.float32)
        top = np.argpartition(raw, -top_k)[-top_k:]
        ranked = top[np.argsort(raw[top])[::-1]]
    return [(int(doc_id), float(scores[doc_id])) for doc_id in ranked[:top_k]]


def make_protocol_subset(semantic_ranked, lexical_ranked, corpus_limit, protocol_docs):
    if protocol_docs >= corpus_limit:
        return list(range(corpus_limit))
    ordered = []
    seen = set()
    for doc_id in [doc_id for doc_id, _ in semantic_ranked] + [doc_id for doc_id, _ in lexical_ranked]:
        if doc_id < corpus_limit and doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)
    cursor = 0
    while len(ordered) < protocol_docs and cursor < corpus_limit:
        if cursor not in seen:
            ordered.append(cursor)
            seen.add(cursor)
        cursor += 1
    return ordered[:protocol_docs]


def build_tf_matrix(subset_terms, query_terms):
    token_to_id = {term: index for index, term in enumerate(dict.fromkeys(query_terms))}
    tf = torch.zeros(len(token_to_id), len(subset_terms), dtype=torch.float32)
    for doc_pos, counter in enumerate(subset_terms):
        for term, token_id in token_to_id.items():
            tf[token_id, doc_pos] = float(counter.get(term, 0))
    query_tokens = torch.tensor([token_to_id[term] for term in query_terms if term in token_to_id], dtype=torch.long)
    lengths = torch.tensor([sum(counter.values()) or 1 for counter in subset_terms], dtype=torch.float32)
    return tf, query_tokens, lengths


def run_protocol_query(
    *,
    query_embedding,
    query_terms,
    subset_doc_ids,
    gold_ids,
    chunk_embeddings,
    chunks,
    term_frequencies,
    config,
    args,
):
    local_gold = {subset_doc_ids.index(doc_id) for doc_id in gold_ids if doc_id in subset_doc_ids}
    doc_embeddings = torch.tensor(np.asarray(chunk_embeddings[subset_doc_ids]), dtype=torch.float32)
    payload = torch.tensor(subset_doc_ids, dtype=torch.float32).unsqueeze(-1)
    query_tensor = torch.tensor(np.asarray(query_embedding), dtype=torch.float32).unsqueeze(0)

    p3_server = Protocol3Server(
        okvs=BinaryOKVS(expansion=3.0, seed=b"real-dataset-p3-okvs"),
        he=AdditivePaillier(key_size=args.he_key_size),
        threshold=args.p3_threshold,
        projection_count=args.p3_projections,
        seed=b"real-dataset-p3",
        simhash_bits=args.simhash_bits,
    )
    p1_server = Protocol1Server(config=config, protocol3=p3_server)
    p1_client = Protocol1Client(config=config, protocol3=Protocol3Client(shuffle_seed=b"real-dataset-p3-client"))
    p3_setup = p1_server.build_filter_setup(doc_embeddings, chunks=subset_doc_ids)
    p3_message = p1_client.make_filter_query(query_tensor, p3_setup)
    candidates = p1_server.recover_candidates(p3_message, num_docs=len(subset_doc_ids))
    semantic = protocol1_finish_from_candidate_mask(
        query_tensor,
        doc_embeddings,
        candidate_mask=candidates.candidate_mask,
        top_k=config.top_k,
        document_payload_shares=payload,
    )
    semantic_ids = [int(value.item()) for value in semantic.documents.reshape(-1)]

    subset_terms = [term_frequencies[doc_id] for doc_id in subset_doc_ids]
    tf_by_token_doc, query_token_ids, lengths = build_tf_matrix(subset_terms, query_terms)
    if query_token_ids.numel() == 0:
        lexical_ids = subset_doc_ids[: config.top_k]
        bm25_scores = torch.zeros(len(subset_doc_ids), dtype=torch.float32)
    else:
        params = DHOPRFParams()
        okvs = BinaryOKVS(expansion=2.4, seed=b"real-dataset-p4-okvs")
        p4_server = Protocol4Server(okvs=okvs, oprf_server=DHOPRFServer(params=params, secret_key=97531))
        p4_client = Protocol4Client(okvs=okvs, oprf_client=DHOPRFClient(params=params))
        p4_setup = p4_server.build_setup(tf_by_token_doc)
        p4_request = p4_client.make_query(query_token_ids)
        p4_response = p4_server.evaluate_oprf(p4_request)
        recovered_tf = p4_client.recover_term_frequencies(p4_response, p4_setup)
        expected_tf = tf_by_token_doc[query_token_ids].T.contiguous()
        if not torch.equal(recovered_tf, expected_tf):
            raise AssertionError("Protocol 4 recovered TF does not match real-data TF baseline")
        bm25 = bm25_components_from_tf(
            recovered_tf,
            lengths,
            num_documents=recovered_tf.shape[0],
            average_document_length=default_average_length(lengths),
            k1=config.bm25_k1,
            b=config.bm25_b,
        )
        bm25_scores = bm25.scores
        retriever = PiscesRetriever(config)
        _, lexical_indicators, _ = retriever.lexical_path_from_scores(bm25_scores)
        lexical_pir = suda_pir_to_share(lexical_indicators, payload)
        lexical_ids = [int(value.item()) for value in lexical_pir.records.reshape(-1)]

    fused_ids = []
    for doc_id in semantic_ids + lexical_ids:
        if doc_id not in fused_ids:
            fused_ids.append(doc_id)
    return {
        "local_gold": local_gold,
        "candidate_size": int(candidates.candidate_mask.sum().item()),
        "semantic_ids": semantic_ids,
        "lexical_ids": lexical_ids,
        "fused_ids": fused_ids[: config.top_k],
        "bm25_scores": bm25_scores,
        "subset_doc_ids": subset_doc_ids,
        "gold_titles": [chunks[doc_id].source for doc_id in gold_ids],
    }


def has_hit(ids, gold_ids):
    return int(any(doc_id in gold_ids for doc_id in ids))


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def top_semantic_by_embedding(chunk_embeddings, query_embedding, corpus_limit, top_k, device):
    if device.type == "cuda":
        chunks = torch.as_tensor(np.array(chunk_embeddings[:corpus_limit], copy=True), dtype=torch.float32, device=device)
        query = torch.as_tensor(np.array(query_embedding, copy=True), dtype=torch.float32, device=device)
        scores = chunks @ query
        values, indices = torch.topk(scores, k=min(top_k, corpus_limit))
        return [(int(index), float(value)) for index, value in zip(indices.cpu().tolist(), values.cpu().tolist())]
    corpus_matrix = np.asarray(chunk_embeddings[:corpus_limit])
    semantic_scores = corpus_matrix @ np.asarray(query_embedding)
    ranked = np.argsort(semantic_scores)[::-1][: min(top_k, corpus_limit)]
    return [(int(doc_id), float(semantic_scores[doc_id])) for doc_id in ranked]


def main():
    args = parse_args()
    started = time.perf_counter()
    device = resolve_device(args.device)
    prepared, chunk_embeddings, query_embeddings, term_frequencies = load_inputs(args)
    tokenizer = QueryTokenizer()
    eval_corpus_limit = min(args.corpus_limit, args.protocol_docs) if args.setup_mode == "offline" else args.corpus_limit
    queries = choose_queries(prepared, query_embeddings.shape[0], eval_corpus_limit, args.limit_queries)
    skipped_queries = 0
    if args.resume_report:
        completed = completed_query_indices(args.report_jsonl)
        before = len(queries)
        queries = [(query_index, query, gold_ids) for query_index, query, gold_ids in queries if query_index not in completed]
        skipped_queries = before - len(queries)
        if not queries:
            section("Real dataset Pisces evaluation")
            print(f"dataset={args.dataset}")
            print(f"report_jsonl={args.report_jsonl}")
            print(f"resume_report=true skipped_queries={skipped_queries}")
            print("No remaining queries to run.")
            return
    corpus_terms = term_frequencies[: args.corpus_limit]
    bm25, vocabulary = build_bm25_vocab(corpus_terms)
    config = PiscesConfig(top_k=args.top_k, simhash_bits=args.simhash_bits, hamming_threshold=args.p3_threshold)

    section("Real dataset Pisces evaluation")
    print(f"dataset={args.dataset}")
    print(f"dataset_path={args.dataset_path or DATASETS[args.dataset]['path']}")
    print(f"chunks_total={len(prepared.chunks):,}, queries_total={len(prepared.queries):,}")
    print(f"query_embeddings_available={query_embeddings.shape[0]:,}")
    print(f"corpus_limit={args.corpus_limit}, protocol_docs={args.protocol_docs}, eval_queries={len(queries)}")
    if args.resume_report:
        print(f"resume_report=true skipped_queries={skipped_queries} report_jsonl={args.report_jsonl}")
    print(f"query_tokenizer={tokenizer.mode}")
    print(f"torch_device={device}")
    print(f"setup_mode={args.setup_mode}")
    if args.protocol_docs < args.corpus_limit:
        if args.setup_mode == "offline":
            print("[Scope] offline protocol setup runs on the first protocol_docs chunks of the selected corpus prefix.")
        else:
            print("[Scope] protocol runs on a semantic/lexical preselected pool without gold injection, not the full corpus_limit.")
    else:
        print("[Scope] protocol runs on the whole selected corpus prefix.")
    print(f"top_k={args.top_k}, simhash_bits={args.simhash_bits}, p3_projections={args.p3_projections}")

    offline_index = None
    if args.setup_mode == "offline":
        offline_doc_count = min(args.protocol_docs, args.corpus_limit)
        section("Offline setup")
        cache_hit = False
        cache_path, cache_fingerprint = setup_cache_path(args, vocabulary, term_frequencies)
        if not args.no_setup_cache and cache_path.exists():
            try:
                offline_index = load_offline_index_from_cache(cache_path)
                cache_hit = True
                print(f"setup_cache=hit path={cache_path}")
            except Exception as exc:
                print(f"setup_cache=unusable path={cache_path} reason={exc}")
        if offline_index is None:
            print(f"setup_cache=miss path={cache_path}")
            offline_index = OfflinePiscesIndex(
                doc_ids=list(range(offline_doc_count)),
                chunk_embeddings=chunk_embeddings,
                term_frequencies=term_frequencies,
                vocabulary=vocabulary,
                config=config,
                args=args,
            )
            if not args.no_setup_cache:
                save_offline_index_to_cache(cache_path, offline_index, cache_fingerprint)
                print(f"setup_cache=saved path={cache_path}")
        print(
            f"P3 setup: docs={offline_doc_count}, okvs_slots={offline_index.p3_setup.table.size}, "
            f"bucket_capacity={offline_index.p3_setup.bucket_capacity}, "
            f"time_s={0.0 if cache_hit else offline_index.p3_setup_s:.3f}{' (cached)' if cache_hit else ''}"
        )
        print(
            f"P4 setup: entries={offline_index.p4_entries:,}, okvs_slots={offline_index.p4_setup.table.size}, "
            f"value_size={offline_index.p4_setup.table.value_size}, "
            f"time_s={0.0 if cache_hit else offline_index.p4_setup_s:.3f}{' (cached)' if cache_hit else ''}"
        )
        print(
            f"setup_wire_bytes: p3={offline_index.p3_setup_wire_bytes:,}, "
            f"p4={offline_index.p4_setup_wire_bytes:,}, "
            f"total={offline_index.p3_setup_wire_bytes + offline_index.p4_setup_wire_bytes:,}"
        )

    counters = Counter()
    candidate_sizes = []
    timing_totals = Counter()
    wire_totals = Counter()
    per_query_records = []
    for eval_index, (query_index, query_text, gold_ids) in enumerate(queries, start=1):
        query_embedding = query_embeddings[query_index]
        query_terms = tokenizer.tokenize(query_text)
        if offline_index is not None:
            subset_doc_ids = offline_index.doc_ids
            result = offline_index.query(query_embedding, query_terms)
        else:
            semantic_ranked = top_semantic_by_embedding(
                chunk_embeddings,
                query_embedding,
                args.corpus_limit,
                max(args.protocol_docs, args.top_k),
                device,
            )
            lexical_ranked = lexical_pool(query_terms, bm25, vocabulary, corpus_terms, top_k=max(args.protocol_docs, args.top_k))
            subset_doc_ids = make_protocol_subset(
                semantic_ranked,
                lexical_ranked,
                args.corpus_limit,
                args.protocol_docs,
            )
            result = run_protocol_query(
                query_embedding=query_embedding,
                query_terms=query_terms,
                subset_doc_ids=subset_doc_ids,
                gold_ids=gold_ids,
                chunk_embeddings=chunk_embeddings,
                chunks=prepared.chunks,
                term_frequencies=term_frequencies,
                config=config,
                args=args,
            )
        result["gold_titles"] = [prepared.chunks[doc_id].source for doc_id in gold_ids]
        result["subset_doc_ids"] = subset_doc_ids
        candidate_sizes.append(result["candidate_size"])
        counters["semantic"] += has_hit(result["semantic_ids"], gold_ids)
        counters["lexical"] += has_hit(result["lexical_ids"], gold_ids)
        counters["dual_union"] += has_hit(result["semantic_ids"] + result["lexical_ids"], gold_ids)
        counters["fused"] += has_hit(result["fused_ids"], gold_ids)
        for name, value in result.get("timings", {}).items():
            timing_totals[name] += float(value)
        for name, value in result.get("wire", {}).items():
            wire_totals[name] += int(value)

        query_record = {
            "dataset": args.dataset,
            "query_index": query_index,
            "query": query_text,
            "gold_ids": sorted(gold_ids),
            "gold_titles": result["gold_titles"],
            "protocol_subset_size": len(subset_doc_ids),
            "candidate_size": result["candidate_size"],
            "semantic_ids": result["semantic_ids"],
            "lexical_ids": result["lexical_ids"],
            "fused_ids": result["fused_ids"],
            "query_token_count": result.get("query_token_count"),
            "query_vocab_hit_count": result.get("query_vocab_hit_count"),
            "timings": result.get("timings", {}),
            "wire": result.get("wire", {}),
        }
        per_query_records.append(query_record)
        append_jsonl(args.report_jsonl, {"type": "query", **query_record})

        if eval_index <= args.verbose_queries:
            print("\n" + "-" * 96)
            print(f"query#{query_index}: {query_text}")
            print(f"gold_ids={sorted(gold_ids)} gold_titles={result['gold_titles']}")
            print(f"protocol_subset_size={len(subset_doc_ids)} candidate_size={result['candidate_size']}")
            print(f"semantic_ids={result['semantic_ids']}")
            print(f"lexical_ids={result['lexical_ids']}")
            print(f"fused_ids={result['fused_ids']}")
            if "timings" in result:
                compact_timing = ", ".join(f"{name}={value:.4f}s" for name, value in result["timings"].items())
                print(f"timings: {compact_timing}")
            if "wire" in result:
                print(f"query_wire_bytes={result['wire']}")

    total = len(queries)
    section("Real-data metrics")
    print(f"processed_queries={total}")
    print(f"semantic_candidate_avg={sum(candidate_sizes) / len(candidate_sizes):.2f}")
    print(f"top_{args.top_k}_semantic_hit={counters['semantic'] / total:.4f}")
    print(f"top_{args.top_k}_lexical_hit={counters['lexical'] / total:.4f}")
    print(f"top_{args.top_k}_dual_union_hit={counters['dual_union'] / total:.4f}")
    print(f"top_{args.top_k}_fused_hit={counters['fused'] / total:.4f}")
    if timing_totals:
        print("timing_totals_s=" + json.dumps({name: round(value, 6) for name, value in timing_totals.items()}, sort_keys=True))
        print("timing_avg_s=" + json.dumps({name: round(value / total, 6) for name, value in timing_totals.items()}, sort_keys=True))
    if wire_totals:
        setup_wire = 0
        if offline_index is not None:
            setup_wire = offline_index.p3_setup_wire_bytes + offline_index.p4_setup_wire_bytes
        print("online_wire_totals_bytes=" + json.dumps(dict(wire_totals), sort_keys=True))
        print(f"offline_setup_wire_bytes={setup_wire}")
        print(f"online_wire_avg_bytes={sum(wire_totals.values()) / total:.2f}")
    print(f"elapsed_s={time.perf_counter() - started:.3f}")
    summary = {
        "type": "summary",
        "dataset": args.dataset,
        "corpus_limit": args.corpus_limit,
        "protocol_docs": args.protocol_docs,
        "processed_queries": total,
        "top_k": args.top_k,
        "hits": {
            "semantic": counters["semantic"] / total,
            "lexical": counters["lexical"] / total,
            "dual_union": counters["dual_union"] / total,
            "fused": counters["fused"] / total,
        },
        "semantic_candidate_avg": sum(candidate_sizes) / len(candidate_sizes),
        "timing_totals_s": dict(timing_totals),
        "wire_totals_bytes": dict(wire_totals),
        "offline_setup_wire_bytes": (
            offline_index.p3_setup_wire_bytes + offline_index.p4_setup_wire_bytes if offline_index is not None else 0
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    append_jsonl(args.report_jsonl, summary)


if __name__ == "__main__":
    main()
