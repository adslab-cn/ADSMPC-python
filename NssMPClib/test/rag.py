import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
import sys
import pickle
import time
import json
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# 引入你的环境
from NssMPC.config import DEVICE, SCALE_BIT, float_scale, param_path
from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.application.neural_network.party.neural_network_party import NeuralNetworkCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.application.neural_network.utils.converter import share_model, load_model, share_data
from NssMPC.crypto.aux_parameter import AssMulTriples, DivKey, GeLUKey, Wrap, SigmaDICFKey, ReciprocalSqrtKey, TanhKey,MatmulTriples,B2AKey
from NssMPC.application.neural_network.layers.mha import SecBertModel
from NssMPC.application.rag.pisces import PiscesConfig
from NssMPC.application.rag.pisces.ops import (
    bm25_components_from_tf,
    default_average_length,
)
from NssMPC.application.rag.pisces.protocol1 import Protocol1Client, Protocol1Server
from NssMPC.application.rag.pisces.protocol2 import Protocol2Client, Protocol2Server

# ==========================================
# 1. 全局配置
# ==========================================
RAG_PROFILES = {
    "squad_dev_q8": {
        "PISCES_RAG_REAL_DATASET": "1",
        "PISCES_RAG_DATASET": "squad_dev_v2",
        "PISCES_RAG_NUM_DOCS": "1204",
        "PISCES_RAG_TOP_K": "10",
        "PISCES_RAG_QUERY_TERM_LIMIT": "8",
        "PISCES_RAG_REQUIRE_QUERY_TERMS": "1",
        "PISCES_RAG_DOC_LEN": "8",
        "PISCES_RAG_P3_PAILLIER_KEY_SIZE": "64",
        "PISCES_RAG_P3_SIMHASH_BITS": "64",
        "PISCES_RAG_P3_THRESHOLD": "14",
        "PISCES_RAG_P3_PROJECTIONS": "160",
    },
    "hotpot_dev_distractor_q8": {
        "PISCES_RAG_REAL_DATASET": "1",
        "PISCES_RAG_DATASET": "hotpotqa_dev_distractor",
        "PISCES_RAG_NUM_DOCS": "269602",
        "PISCES_RAG_TOP_K": "10",
        "PISCES_RAG_QUERY_TERM_LIMIT": "8",
        "PISCES_RAG_REQUIRE_QUERY_TERMS": "1",
        "PISCES_RAG_DOC_LEN": "8",
        "PISCES_RAG_P3_PAILLIER_KEY_SIZE": "64",
        "PISCES_RAG_P3_SIMHASH_BITS": "64",
        "PISCES_RAG_P3_THRESHOLD": "14",
        "PISCES_RAG_P3_PROJECTIONS": "160",
    },
}
RAG_PROFILE = os.environ.get("PISCES_RAG_PROFILE", "")
if RAG_PROFILE:
    if RAG_PROFILE not in RAG_PROFILES:
        raise ValueError(f"unknown PISCES_RAG_PROFILE={RAG_PROFILE}; choices={sorted(RAG_PROFILES)}")
    for _profile_key, _profile_value in RAG_PROFILES[RAG_PROFILE].items():
        os.environ.setdefault(_profile_key, _profile_value)

BERT_CONFIG = {
    "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
    "intermediate_size": 512, "vocab_size": 30522, 
    "max_position_embeddings": 512, "type_vocab_size": 2
}
BATCH = 1
SEQ = 8
NUM_DOCS = int(os.environ.get("PISCES_RAG_NUM_DOCS", "10"))  # 知识库的文档库大小
TOP_K = int(os.environ.get("PISCES_RAG_TOP_K", "1"))      # 我们想要召回的文档数量
QUERY_LEN = 8
SEM_DOC_LEN = int(os.environ.get("PISCES_RAG_DOC_LEN", "24"))  # 语义路召回的文档长度
LEX_DOC_LEN = SEM_DOC_LEN  # BM25路召回的文档长度
# 最终送入模型的总长度 = Query + Doc1(语义) + Doc2(词汇) = 56
TOTAL_SEQ = QUERY_LEN + SEM_DOC_LEN + LEX_DOC_LEN
VOCAB_SIZE_BM25 = 100
DEBUG = False
QUERY_BM25_TOKENS = torch.tensor([5, 8])
P3_SIMHASH_BITS = int(os.environ.get("PISCES_RAG_P3_SIMHASH_BITS", str(PiscesConfig().simhash_bits)))
P3_THRESHOLD = int(os.environ.get("PISCES_RAG_P3_THRESHOLD", "2"))
P3_PROJECTION_COUNT = int(os.environ.get("PISCES_RAG_P3_PROJECTIONS", "8"))
P3_PAILLIER_KEY_SIZE = int(os.environ.get("PISCES_RAG_P3_PAILLIER_KEY_SIZE", "256"))
RAG_TOPK_PENALTY = float(os.environ.get("PISCES_RAG_TOPK_PENALTY", "-1000000.0"))
P4_OKVS_EXPANSION = float(os.environ.get("PISCES_RAG_P4_OKVS_EXPANSION", "2.4"))
P4_OPRF_SECRET_KEY = int(os.environ.get("PISCES_RAG_P4_OPRF_SECRET_KEY", "97531"))
RAG_CONFIG = PiscesConfig(top_k=TOP_K, simhash_bits=P3_SIMHASH_BITS, hamming_threshold=P3_THRESHOLD)
REAL_DATASET_MODE = os.environ.get("PISCES_RAG_REAL_DATASET") == "1"
REAL_DATASET_NAME = os.environ.get("PISCES_RAG_DATASET", "squad_dev_v2")
REAL_QUERY_INDEX = int(os.environ.get("PISCES_RAG_QUERY_INDEX", "0"))
REAL_QUERY_OFFSET = int(os.environ.get("PISCES_RAG_QUERY_OFFSET", "0"))
REAL_QUERY_COUNT = int(os.environ.get("PISCES_RAG_QUERY_COUNT", "1"))
REAL_QUERY_TERM_LIMIT = int(os.environ.get("PISCES_RAG_QUERY_TERM_LIMIT", "0"))
REAL_REQUIRE_QUERY_TERMS = os.environ.get("PISCES_RAG_REQUIRE_QUERY_TERMS") == "1"
REAL_REPORT_JSON = os.environ.get("PISCES_RAG_REPORT_JSON", "")
REAL_CACHE_DIR = Path(os.environ.get("PISCES_RAG_CACHE_DIR", "/home/adslab/pazika/pisces/.embedding-cache"))
SETUP_CACHE_ENABLED = os.environ.get("PISCES_RAG_SETUP_CACHE", "1") != "0"
SETUP_CACHE_DIR = Path(os.environ.get("PISCES_RAG_SETUP_CACHE_DIR", "data/pisces_setup_cache/rag"))
REAL_SQUAD_PATH = Path(os.environ.get("PISCES_RAG_SQUAD_PATH", "data/squad/dev-v2.0.json"))
REAL_HOTPOT_PATH = Path(
    os.environ.get(
        "PISCES_RAG_HOTPOT_PATH",
        "/home/adslab/pazika/pisces/hotpot/hotpot_dev_distractor_v1.json",
    )
)
BERT_WEIGHTS_PATH = Path(os.environ.get("PISCES_RAG_BERT_WEIGHTS", "NssMPClib/test/bert_tiny_weights.pth"))
LOAD_BERT_WEIGHTS = os.environ.get("PISCES_RAG_LOAD_BERT_WEIGHTS", "1") != "0"
REAL_INPUTS = None
VERBOSE = os.environ.get("PISCES_RAG_VERBOSE") == "1"
PROTOCOL_AUDIT = os.environ.get("PISCES_RAG_PROTOCOL_AUDIT") == "1"
AUDIT_SAMPLE = int(os.environ.get("PISCES_RAG_AUDIT_SAMPLE", "8"))
SUPPRESS_NATIVE_LOGS = os.environ.get("PISCES_RAG_NATIVE_LOGS") != "1"
PANTHER_GC_TOPK_BIN = Path(
    os.environ.get("PANTHER_GC_TOPK_BIN", "/tmp/OpenPanther/bazel-bin/experimental/panther/pisces_gc_topk_cli")
)
_TOKENIZER_LOCK = threading.Lock()
_REAL_CONTEXT_LOCK = threading.Lock()
ASS_MUL_TRIPLE_COUNT = int(os.environ.get("PISCES_RAG_ASS_MUL_TRIPLE_COUNT", "50000000"))
DIV_KEY_COUNT = int(os.environ.get("PISCES_RAG_DIV_KEY_COUNT", "100000"))
DIV_KEY_SAVED_NAME = os.environ.get("PISCES_RAG_DIV_KEY_SAVED_NAME", "DivKey")
GELU_KEY_COUNT = int(os.environ.get("PISCES_RAG_GELU_KEY_COUNT", "100000"))
TANH_KEY_COUNT = int(os.environ.get("PISCES_RAG_TANH_KEY_COUNT", "100000"))
WRAP_KEY_COUNT = int(os.environ.get("PISCES_RAG_WRAP_KEY_COUNT", "10000000"))
RECIPROCAL_SQRT_KEY_COUNT = int(os.environ.get("PISCES_RAG_RECIPROCAL_SQRT_KEY_COUNT", "10000"))
SIGMA_DICF_KEY_COUNT = int(os.environ.get("PISCES_RAG_SIGMA_DICF_KEY_COUNT", "100000"))
B2A_KEY_COUNT = int(os.environ.get("PISCES_RAG_B2A_KEY_COUNT", "1000000"))
B2A_KEY_SAVED_NAME = os.environ.get("PISCES_RAG_B2A_KEY_SAVED_NAME", "B2AKey")
AUTO_GEN_PARAMS = os.environ.get("PISCES_RAG_AUTO_GEN_PARAMS", "1") != "0"
CHECK_PARAMS = os.environ.get("PISCES_RAG_CHECK_PARAMS", "1") != "0"
PARAMS_ONLY = os.environ.get("PISCES_RAG_PARAMS_ONLY", "0") == "1"
PARAM_SAFETY_FACTOR = float(os.environ.get("PISCES_RAG_PARAM_SAFETY_FACTOR", "1.10"))
OBLIVIOUS_FILTER_STAGE = "Oblivious-Filter"
MULTLPSI_STAGE = "MultLPSI"
RUN_METRICS = {
    "profile": RAG_PROFILE or None,
    "config": {},
    "data": {},
    "timings": {},
    "checkpoints": {},
    "communication": {},
    "retrieval": {},
    "pir": {},
    "final": {},
}
RUN_METRICS_LOCK = threading.Lock()


def log(role, stage, message):
    print(f"[{role}][{stage}] {message}", flush=True)


def update_metrics(section, values):
    with RUN_METRICS_LOCK:
        RUN_METRICS.setdefault(section, {}).update(values)


# Runtime reporting helpers. These functions only control logs and JSON reports;
# the Pisces protocol logic below does not depend on them for correctness.
def debug(role, stage, message):
    if VERBOSE:
        log(role, stage, message)


def audit_log(role, stage, message):
    if PROTOCOL_AUDIT:
        log(role, f"Audit:{stage}", message)


def load_bert_tiny_weights(model, role):
    if not LOAD_BERT_WEIGHTS:
        log(role, "Setup-model", "PISCES_RAG_LOAD_BERT_WEIGHTS=0, using zero-initialized SecBertModel weights")
        update_metrics("model", {"weights_loaded": False, "weights_path": None})
        return model
    if not BERT_WEIGHTS_PATH.exists():
        log(role, "Setup-model", f"weights file not found: {BERT_WEIGHTS_PATH}; using zero-initialized weights")
        update_metrics("model", {"weights_loaded": False, "weights_path": str(BERT_WEIGHTS_PATH)})
        return model

    state_dict = torch.load(BERT_WEIGHTS_PATH, map_location=DEVICE)
    state_dict.pop("embeddings.position_ids", None)
    normalized_state = {}
    for key, value in state_dict.items():
        normalized_state[key[5:] if key.startswith("bert.") else key] = value

    loaded = 0
    missing = []
    for name, param in model.named_parameters():
        if name in normalized_state:
            param.data = normalized_state[name].to(DEVICE).to(torch.float32)
            loaded += 1
        else:
            missing.append(name)

    sample_norms = {}
    for sample_name in (
        "embeddings.word_embeddings.weight",
        "encoder.layer.0.intermediate.dense.weight",
        "pooler.dense.weight",
    ):
        param = dict(model.named_parameters()).get(sample_name)
        if param is not None:
            sample_norms[sample_name] = float(param.detach().abs().sum().item())
    log(
        role,
        "Setup-model",
        f"loaded BERT weights from {BERT_WEIGHTS_PATH}, loaded_params={loaded}, "
        f"missing_params={len(missing)}, sample_abs_sums={sample_norms}",
    )
    if missing:
        debug(role, "Setup-model", f"missing weight names sample={missing[:8]}")
    update_metrics(
        "model",
        {
            "weights_loaded": True,
            "weights_path": str(BERT_WEIGHTS_PATH),
            "loaded_params": int(loaded),
            "missing_params": int(len(missing)),
            "sample_abs_sums": sample_norms,
        },
    )
    return model


def _comm_snapshot(role):
    party = globals().get(str(role).lower())
    communicator = getattr(party, "communicator", None)
    if communicator is None:
        return None
    return {
        "send_rounds": int(communicator.comm_rounds.get("send", 0)),
        "recv_rounds": int(communicator.comm_rounds.get("recv", 0)),
        "send_bytes": int(communicator.comm_bytes.get("send", 0)),
        "recv_bytes": int(communicator.comm_bytes.get("recv", 0)),
    }


def _comm_delta(start, end):
    if start is None or end is None:
        return None
    return {key: int(end[key] - start[key]) for key in start}


def _zero_comm():
    return {"send_rounds": 0, "recv_rounds": 0, "send_bytes": 0, "recv_bytes": 0}


def _add_comm(total, delta):
    if delta is None:
        return total
    for key in ("send_rounds", "recv_rounds", "send_bytes", "recv_bytes"):
        total[key] = int(total.get(key, 0)) + int(delta.get(key, 0))
    return total


def _directional_comm(communication, stage):
    server = communication.get(f"Server.{stage}", _zero_comm())
    client = communication.get(f"Client.{stage}", _zero_comm())
    upload = max(int(client.get("send_bytes", 0)), int(server.get("recv_bytes", 0)))
    download = max(int(server.get("send_bytes", 0)), int(client.get("recv_bytes", 0)))
    return {
        "upload_bytes": int(upload),
        "download_bytes": int(download),
        "total_directional_bytes": int(upload + download),
        "upload_mb": _mb(upload),
        "download_mb": _mb(download),
        "total_directional_mb": _mb(upload + download),
        "client_send_bytes": int(client.get("send_bytes", 0)),
        "client_recv_bytes": int(client.get("recv_bytes", 0)),
        "server_send_bytes": int(server.get("send_bytes", 0)),
        "server_recv_bytes": int(server.get("recv_bytes", 0)),
        "client_send_rounds": int(client.get("send_rounds", 0)),
        "client_recv_rounds": int(client.get("recv_rounds", 0)),
        "server_send_rounds": int(server.get("send_rounds", 0)),
        "server_recv_rounds": int(server.get("recv_rounds", 0)),
    }


def _mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


class CheckpointReporter:
    """Stage reporter for the main protocol flow.

    The protocol code calls mark() after a stage completes. This keeps timing
    and communication reporting out of the stage body, so the surrounding code
    reads in the same order as the Pisces protocol.
    """

    def __init__(self, role, run_start=None):
        self.role = role
        self.run_start = run_start if run_start is not None else time.perf_counter()
        self.last_time = self.run_start
        self.last_comm = _comm_snapshot(role)

    def mark(self, stage):
        now = time.perf_counter()
        elapsed = now - self.last_time
        cumulative = now - self.run_start
        comm_now = _comm_snapshot(self.role)
        comm = _comm_delta(self.last_comm, comm_now)

        update_metrics("timings", {f"{self.role}.{stage}": elapsed})
        checkpoint_payload = {
            "elapsed_seconds": float(elapsed),
            "cumulative_seconds": float(cumulative),
        }
        if comm is not None:
            checkpoint_payload["communication"] = comm
            update_metrics("communication", {f"{self.role}.{stage}": comm})
            log(
                self.role,
                "Checkpoint",
                f"{stage}: +{elapsed:.2f}s, total={cumulative:.2f}s, "
                f"send={_mb(comm['send_bytes']):.3f}MB, recv={_mb(comm['recv_bytes']):.3f}MB",
            )
        else:
            log(self.role, "Checkpoint", f"{stage}: +{elapsed:.2f}s, total={cumulative:.2f}s")

        update_metrics("checkpoints", {f"{self.role}.{stage}": checkpoint_payload})
        self.last_time = now
        self.last_comm = comm_now


def finalize_total(role, run_start):
    total = time.perf_counter() - run_start
    prefix = f"{role}."
    with RUN_METRICS_LOCK:
        timings = dict(RUN_METRICS.get("timings", {}))
    covered = sum(
        float(value)
        for key, value in timings.items()
        if key.startswith(prefix)
        and not key.endswith(".Total")
        and not key.endswith(".Unattributed")
    )
    unattributed = max(0.0, total - covered)
    update_metrics(
        "timings",
        {
            f"{role}.Unattributed": unattributed,
            f"{role}.Total": total,
        },
    )
    log(role, "Unattributed", f"{unattributed:.2f}s outside named stages")
    log(role, "Total", f"done in {total:.2f}s")
    return total


# Batch experiment bookkeeping only. The per-query retrieval loop keeps these
# tiny scopes so the report can average each protocol stage across many queries.
@contextmanager
def timed_batch(role, query_no, query_index, stage):
    start = time.perf_counter()
    comm_start = _comm_snapshot(role)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        comm = _comm_delta(comm_start, _comm_snapshot(role))
        with RUN_METRICS_LOCK:
            query_key = str(int(query_no))
            batch = RUN_METRICS.setdefault("batch_timings", {})
            entry = batch.setdefault(
                query_key,
                {"query_no": int(query_no), "query_index": int(query_index)},
            )
            entry[f"{role}.{stage}"] = float(elapsed)
            if comm is not None:
                entry.setdefault("communication", {})[f"{role}.{stage}"] = comm


def update_batch_result(query_no, values):
    with RUN_METRICS_LOCK:
        query_key = str(int(query_no))
        batch = RUN_METRICS.setdefault("batch_timings", {})
        entry = batch.setdefault(query_key, {"query_no": int(query_no)})
        entry.update(values)


def summarize_batch_metrics():
    with RUN_METRICS_LOCK:
        batch = dict(RUN_METRICS.get("batch_timings", {}))
    if not batch:
        return
    stage_values = {}
    stage_communication = {}
    native_communication = {
        "semantic_topk_gc_bytes": 0,
        "lexical_topk_gc_bytes": 0,
        "topk_gc_total_bytes": 0,
        "semantic_pir_query_bytes": 0,
        "semantic_pir_response_bytes": 0,
        "lexical_pir_query_bytes": 0,
        "lexical_pir_response_bytes": 0,
        "suda_pir_query_bytes": 0,
        "suda_pir_response_bytes": 0,
    }
    hit_counts = {"semantic_hit": 0, "lexical_hit": 0, "dual_union_hit": 0}
    query_count = 0
    for entry in batch.values():
        query_count += 1
        for key, value in entry.items():
            if "." in key and isinstance(value, (int, float)):
                stage_values.setdefault(key, []).append(float(value))
        for key, comm in (entry.get("communication") or {}).items():
            total = stage_communication.setdefault(key, _zero_comm())
            _add_comm(total, comm)
        for key in native_communication:
            native_communication[key] += int(entry.get(key, 0) or 0)
        for hit_key in hit_counts:
            if entry.get(hit_key):
                hit_counts[hit_key] += 1
    averages = {
        key: float(sum(values) / len(values))
        for key, values in sorted(stage_values.items())
        if values
    }
    update_metrics(
        "batch_summary",
        {
            "query_count": int(query_count),
            "stage_average_seconds": averages,
            "stage_communication": stage_communication,
            "native_communication": native_communication,
            "semantic_hit_rate": hit_counts["semantic_hit"] / query_count if query_count else 0.0,
            "lexical_hit_rate": hit_counts["lexical_hit"] / query_count if query_count else 0.0,
            "dual_union_hit_rate": hit_counts["dual_union_hit"] / query_count if query_count else 0.0,
        },
    )


def print_pisces_contract(role):
    log(
        role,
        "Config",
        f"docs={NUM_DOCS}, top_k={TOP_K}, doc_len={SEM_DOC_LEN}, "
        f"real_dataset={REAL_DATASET_MODE}, dataset={REAL_DATASET_NAME}, "
        f"profile={RAG_PROFILE or 'custom'}, query_count={REAL_QUERY_COUNT}, "
        f"query_offset={REAL_QUERY_OFFSET}, "
        f"query_term_limit={REAL_QUERY_TERM_LIMIT or 'all'}, "
        f"require_query_terms={REAL_REQUIRE_QUERY_TERMS}, verbose={VERBOSE}, "
        f"protocol_audit={PROTOCOL_AUDIT}, native_logs={not SUPPRESS_NATIVE_LOGS}, "
        f"p3_simhash_bits={P3_SIMHASH_BITS}, p3_projections={P3_PROJECTION_COUNT}, "
        f"p4_okvs_expansion={P4_OKVS_EXPANSION}, "
        f"setup_cache={SETUP_CACHE_ENABLED}, payload_pir_scope=full-db",
    )


def _plain_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, RingTensor):
        return value.convert_to_real_field().detach().cpu()
    if isinstance(value, ArithmeticSecretSharing):
        return value.item.convert_to_real_field().detach().cpu()
    try:
        return torch.as_tensor(value).detach().cpu()
    except Exception:
        return None


def _sample_list(value, sample=AUDIT_SAMPLE):
    tensor = _plain_tensor(value)
    if tensor is None:
        return repr(value)
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return []
    return flat[: min(sample, flat.numel())].tolist()


def _value_shape(value):
    return tuple(int(dim) for dim in value.shape) if hasattr(value, "shape") else None


def _is_secret_value(value):
    return isinstance(value, ArithmeticSecretSharing)


# Protocol audit helpers. They deliberately restore shares only when
# PISCES_RAG_PROTOCOL_AUDIT=1, so normal benchmark runs stay on the protocol path.
def _max_abs_err(left, right):
    left_t = _plain_tensor(left)
    right_t = _plain_tensor(right)
    if left_t is None or right_t is None:
        return None
    if left_t.shape != right_t.shape:
        return f"shape mismatch {tuple(left_t.shape)} != {tuple(right_t.shape)}"
    if left_t.numel() == 0:
        return 0.0
    return float((left_t.float() - right_t.float()).abs().max().item())


def audit_value(role, stage, name, value, *, expected=None, plaintext_visibility="audit/local"):
    if not PROTOCOL_AUDIT:
        return
    msg = (
        f"{name}: type={type(value).__name__}, shape={_value_shape(value)}, "
        f"secret={_is_secret_value(value)}, visibility={plaintext_visibility}, sample={_sample_list(value)}"
    )
    if expected is not None:
        msg += f", expected_sample={_sample_list(expected)}, max_abs_err={_max_abs_err(value, expected)}"
    audit_log(role, stage, msg)


def audit_ass_pair(role, party, stage, name, local_share, *, expected=None, plaintext_visibility="secret-shared"):
    if not PROTOCOL_AUDIT:
        return None
    audit_value(role, stage, f"{name}.local_share", local_share, plaintext_visibility=plaintext_visibility)
    if role == "Server":
        remote_share = party.receive()
        party.send(local_share)
    else:
        party.send(local_share)
        remote_share = party.receive()
    restored = ArithmeticSecretSharing.restore_from_shares(local_share, remote_share).convert_to_real_field()
    audit_value(
        role,
        stage,
        f"{name}.restored_plaintext_for_audit",
        restored,
        expected=expected,
        plaintext_visibility="restored only because PISCES_RAG_PROTOCOL_AUDIT=1",
    )
    return restored


def expected_semantic_scores(query_embedding, db_embeddings, candidate_mask):
    scores = (query_embedding.to(db_embeddings.device) * db_embeddings).sum(dim=-1).reshape(-1)
    return scores + (1.0 - candidate_mask.to(scores.device).float()) * RAG_TOPK_PENALTY


def expected_bm25_from_tf(tf, document_lengths):
    return bm25_components_from_tf(
        tf,
        document_lengths,
        num_documents=int(tf.shape[0]),
        average_document_length=default_average_length(document_lengths),
        k1=RAG_CONFIG.bm25_k1,
        b=RAG_CONFIG.bm25_b,
    )


def indicator_from_ids(ids, k, num_items, *, device=DEVICE):
    indicator = torch.zeros(k, num_items, dtype=torch.float32, device=device)
    for row, index in enumerate(torch.as_tensor(ids, dtype=torch.long).reshape(-1).tolist()[:k]):
        indicator[row, int(index)] = 1.0
    return indicator


def ids_from_indicator(indicator):
    tensor = _plain_tensor(indicator).round()
    return tensor.argmax(dim=1).to(dtype=torch.long)


def print_topk_audit(role, path, audit):
    if audit is None:
        log(role, f"{path}-TopK", "skipped: empty candidate set")
        return
    msg = (
        f"algorithm={audit.algorithm}, backend={audit.backend}, network={audit.network}, "
        f"comparisons={audit.comparisons}, paper_backend_available={audit.paper_backend_available}"
    )
    if audit.gc_communication_bytes is not None:
        msg += f", gc_bytes={audit.gc_communication_bytes}, value_bits={audit.gc_value_bits}, port={audit.gc_port}"
    log(role, f"{path}-TopK", msg)


def print_pir_audit(role, path, pir_or_audit):
    if pir_or_audit is None:
        log(role, f"{path}-PIR", "skipped: empty candidate set")
        return
    audit = pir_or_audit.audit if hasattr(pir_or_audit, "audit") else pir_or_audit
    if isinstance(audit, dict):
        msg = (
            f"implementation={audit['implementation']}, "
            f"paper_backend_available={audit['paper_backend_available']}, "
            f"output_shape={audit['output_shape']}"
        )
        if audit.get("native_batch_size") is not None:
            msg += (
                f", modulus={audit['polynomial_modulus']}, "
                f"batch={audit['native_batch_size']}, padded={audit['native_padded_database_size']}, "
                f"query_bytes={audit['native_query_bytes']}, response_bytes={audit['native_response_bytes']}"
            )
        log(role, f"{path}-PIR", msg)
        debug(role, f"{path}-PIR", f"paper_backend={audit['paper_backend']}, backend_gap={audit['backend_gap']}")
        return
    msg = (
        f"implementation={audit.implementation}, paper_backend_available={audit.paper_backend_available}, "
        f"output_shape={audit.output_shape}"
    )
    if audit.native_batch_size is not None:
        msg += (
            f", modulus={audit.polynomial_modulus}, batch={audit.native_batch_size}, "
            f"padded={audit.native_padded_database_size}, query_bytes={audit.native_query_bytes}, "
            f"response_bytes={audit.native_response_bytes}"
        )
    log(role, f"{path}-PIR", msg)
    debug(role, f"{path}-PIR", f"paper_backend={audit.paper_backend}, backend_gap={audit.backend_gap}")


def pir_audit_to_metrics(audit):
    if audit is None:
        return {
            "implementation": "skipped-empty-candidate-set",
            "paper_backend_available": True,
            "output_shape": (0, SEM_DOC_LEN, BERT_CONFIG["hidden_size"]),
            "query_bytes": 0,
            "response_bytes": 0,
            "padded_database_size": 0,
            "batch_size": 0,
        }
    if isinstance(audit, dict):
        return {
            "implementation": audit.get("implementation"),
            "paper_backend_available": audit.get("paper_backend_available"),
            "output_shape": audit.get("output_shape"),
            "query_bytes": audit.get("query_bytes", audit.get("native_query_bytes")),
            "response_bytes": audit.get("response_bytes", audit.get("native_response_bytes")),
            "padded_database_size": audit.get("padded_database_size", audit.get("native_padded_database_size")),
            "batch_size": audit.get("batch_size", audit.get("native_batch_size")),
        }
    return {
        "implementation": audit.implementation,
        "paper_backend_available": audit.paper_backend_available,
        "output_shape": tuple(int(dim) for dim in audit.output_shape),
        "query_bytes": audit.native_query_bytes,
        "response_bytes": audit.native_response_bytes,
        "padded_database_size": audit.native_padded_database_size,
        "batch_size": audit.native_batch_size,
    }


def topk_audit_to_metrics(audit):
    if audit is None:
        return {
            "algorithm": "skipped-empty-candidate-set",
            "backend": "none",
            "network": "none",
            "num_items": 0,
            "top_k": 0,
            "comparisons": 0,
            "paper_backend_available": True,
            "gc_communication_bytes": 0,
            "gc_value_bits": None,
            "gc_port": None,
        }
    return {
        "algorithm": audit.algorithm,
        "backend": audit.backend,
        "network": audit.network,
        "num_items": audit.num_items,
        "top_k": audit.top_k,
        "comparisons": audit.comparisons,
        "paper_backend_available": audit.paper_backend_available,
        "gc_communication_bytes": audit.gc_communication_bytes,
        "gc_value_bits": audit.gc_value_bits,
        "gc_port": audit.gc_port,
    }


PAPER_EFFICIENCY_REFERENCES = {
    "squad_dev_v2": {
        "semantic_coarse_to_fine": {
            "table": "Table 4",
            "dataset_label": "SQuAD Dev v2.0",
            "time_seconds": 3.41,
            "upload_mb": 21.96,
            "download_mb": 13.93,
            "accuracy_percent": 75.96,
        },
        "semantic_fine_only": {
            "table": "Table 4",
            "dataset_label": "SQuAD Dev v2.0",
            "time_seconds": 1.714,
            "upload_mb": 24.06,
            "download_mb": 24.147,
            "accuracy_percent": 99.64,
        },
        "lexical_multlpsi": {
            "table": "Table 5",
            "dataset_label": "SQuAD Dev v2.0",
            "time_seconds": 0.008,
            "upload_mb": 0.0004,
            "download_mb": 1.49,
        },
        "lexical_labeled_psi": {
            "table": "Table 5",
            "dataset_label": "SQuAD Dev v2.0",
            "time_seconds": 3.15,
            "upload_mb": 0.48,
            "download_mb": 2.03,
        },
    },
    "squad_train_v2": {
        "semantic_coarse_to_fine": {
            "table": "Table 4",
            "dataset_label": "SQuAD Train v2.0",
            "time_seconds": 4.46,
            "upload_mb": 43.51,
            "download_mb": 118.29,
            "accuracy_percent": 74.86,
        },
        "semantic_fine_only": {
            "table": "Table 4",
            "dataset_label": "SQuAD Train v2.0",
            "time_seconds": 3.66,
            "upload_mb": 87.97,
            "download_mb": 315.18,
            "accuracy_percent": 99.95,
        },
        "lexical_multlpsi": {
            "table": "Table 5",
            "dataset_label": "SQuAD Train v2.0",
            "time_seconds": 0.099,
            "upload_mb": 0.0004,
            "download_mb": 22.64,
        },
        "lexical_labeled_psi": {
            "table": "Table 5",
            "dataset_label": "SQuAD Train v2.0",
            "time_seconds": 45.89,
            "upload_mb": 6.99,
            "download_mb": 30.05,
        },
    },
    "hotpot_dev_distractor": {
        "semantic_coarse_to_fine": {
            "table": "Table 4",
            "dataset_label": "HotpotQA Dev distractor",
            "time_seconds": 20.10,
            "upload_mb": 324.90,
            "download_mb": 1439.70,
            "accuracy_percent": 79.80,
        },
        "semantic_fine_only": {
            "table": "Table 4",
            "dataset_label": "HotpotQA Dev distractor",
            "time_seconds": 34.19,
            "upload_mb": 1008.39,
            "download_mb": 4610.26,
            "accuracy_percent": 99.94,
        },
        "lexical_multlpsi": {
            "table": "Table 5",
            "dataset_label": "HotpotQA Dev distractor",
            "time_seconds": 2.35,
            "upload_mb": 0.0006,
            "download_mb": 79.82,
        },
        "lexical_labeled_psi": {
            "table": "Table 5",
            "dataset_label": "HotpotQA Dev distractor",
            "time_seconds": 1051.98,
            "upload_mb": 161.61,
            "download_mb": 382.26,
        },
    },
    "hotpot_dev_fullwiki": {
        "semantic_coarse_to_fine": {
            "table": "Table 4",
            "dataset_label": "HotpotQA Dev fullwiki",
            "time_seconds": 20.76,
            "upload_mb": 330.69,
            "download_mb": 1467.45,
            "accuracy_percent": 78.23,
        },
        "semantic_fine_only": {
            "table": "Table 4",
            "dataset_label": "HotpotQA Dev fullwiki",
            "time_seconds": 33.91,
            "upload_mb": 1031.45,
            "download_mb": 4719.76,
            "accuracy_percent": 99.97,
        },
        "lexical_multlpsi": {
            "table": "Table 5",
            "dataset_label": "HotpotQA Dev fullwiki",
            "time_seconds": 2.59,
            "upload_mb": 0.0006,
            "download_mb": 81.44,
        },
        "lexical_labeled_psi": {
            "table": "Table 5",
            "dataset_label": "HotpotQA Dev fullwiki",
            "time_seconds": 1179.58,
            "upload_mb": 176.89,
            "download_mb": 414.16,
        },
    },
}


def _paper_dataset_key():
    dataset = (REAL_DATASET_NAME or "").lower()
    profile = (RAG_PROFILE or "").lower()
    combined = f"{dataset} {profile}"
    if "squad" in combined and "train" in combined:
        return "squad_train_v2"
    if "squad" in combined:
        return "squad_dev_v2"
    if "hotpot" in combined and "fullwiki" in combined:
        return "hotpot_dev_fullwiki"
    if "hotpot" in combined:
        return "hotpot_dev_distractor"
    return None


def _ratio(actual, reference):
    if reference in (None, 0):
        return None
    return float(actual) / float(reference)


def _stage_seconds(timings, batch_summary, stage):
    averages = batch_summary.get("stage_average_seconds", {}) if isinstance(batch_summary, dict) else {}
    values = []
    for role in ("Client", "Server"):
        key = f"{role}.{stage}"
        if key in averages:
            values.append(float(averages[key]))
        elif key in timings:
            values.append(float(timings[key]))
    if not values:
        return None
    return max(values)


def _native_bytes_from_report(section, key):
    value = section.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def build_paper_comparison_report():
    with RUN_METRICS_LOCK:
        timings = dict(RUN_METRICS.get("timings", {}))
        communication = dict(RUN_METRICS.get("communication", {}))
        batch_summary = dict(RUN_METRICS.get("batch_summary", {}))
        retrieval = dict(RUN_METRICS.get("retrieval", {}))
        pir = dict(RUN_METRICS.get("pir", {}))
        data = dict(RUN_METRICS.get("data", {}))

    if batch_summary.get("stage_communication"):
        communication = dict(batch_summary["stage_communication"])
    query_count = int(batch_summary.get("query_count") or REAL_QUERY_COUNT or 1)
    query_count = max(1, query_count)
    native = dict(batch_summary.get("native_communication", {}))

    if not native:
        semantic_topk = retrieval.get("semantic_topk_audit", {}) if isinstance(retrieval, dict) else {}
        lexical_topk = retrieval.get("lexical_topk_audit", {}) if isinstance(retrieval, dict) else {}
        semantic_pir = pir.get("Client.semantic", pir.get("Server.semantic", {})) if isinstance(pir, dict) else {}
        lexical_pir = pir.get("Client.lexical", pir.get("Server.lexical", {})) if isinstance(pir, dict) else {}
        native = {
            "semantic_topk_gc_bytes": _native_bytes_from_report(semantic_topk, "gc_communication_bytes"),
            "lexical_topk_gc_bytes": _native_bytes_from_report(lexical_topk, "gc_communication_bytes"),
            "semantic_pir_query_bytes": _native_bytes_from_report(semantic_pir, "query_bytes"),
            "semantic_pir_response_bytes": _native_bytes_from_report(semantic_pir, "response_bytes"),
            "lexical_pir_query_bytes": _native_bytes_from_report(lexical_pir, "query_bytes"),
            "lexical_pir_response_bytes": _native_bytes_from_report(lexical_pir, "response_bytes"),
        }
        native["topk_gc_total_bytes"] = native["semantic_topk_gc_bytes"] + native["lexical_topk_gc_bytes"]
        native["suda_pir_query_bytes"] = native["semantic_pir_query_bytes"] + native["lexical_pir_query_bytes"]
        native["suda_pir_response_bytes"] = native["semantic_pir_response_bytes"] + native["lexical_pir_response_bytes"]

    stage_directional = {
        stage: _directional_comm(communication, stage)
        for stage in (
            OBLIVIOUS_FILTER_STAGE,
            MULTLPSI_STAGE,
            "Protocol2-bm25",
            "TopK",
            "Suda-PIR",
        )
    }
    retrieval_upload = sum(item["upload_bytes"] for item in stage_directional.values())
    retrieval_download = sum(item["download_bytes"] for item in stage_directional.values())

    report = {
        "scope_note": (
            "Paper Tables 4 and 5 report each retrieval path separately. This report keeps the same "
            "upload=client-to-server and download=server-to-client convention where the direction is known. "
            "TopK GC native bytes are total two-party GC traffic, so they are reported separately rather "
            "than forced into upload/download."
        ),
        "dataset_key": _paper_dataset_key(),
        "query_count": int(query_count),
        "stages": stage_directional,
        "setup_cache": {
            "p3": data.get("p3_setup_cache"),
            "p4": data.get("p4_setup_cache"),
            "note": (
                "When setup_cache.ready is true, the online socket traffic excludes the cached setup object. "
                "The *_bytes fields are local serialized cache sizes, useful for estimating the one-time "
                "setup transfer that was amortized or skipped in this run."
            ),
        },
        "end_to_end_retrieval_socket": {
            "upload_bytes": int(retrieval_upload),
            "download_bytes": int(retrieval_download),
            "total_directional_bytes": int(retrieval_upload + retrieval_download),
            "upload_mb": _mb(retrieval_upload),
            "download_mb": _mb(retrieval_download),
            "total_directional_mb": _mb(retrieval_upload + retrieval_download),
            "upload_mb_per_query": _mb(retrieval_upload) / query_count,
            "download_mb_per_query": _mb(retrieval_download) / query_count,
            "total_directional_mb_per_query": _mb(retrieval_upload + retrieval_download) / query_count,
        },
        "native_bridges": {
            "topk_gc_total_bytes": int(native.get("topk_gc_total_bytes", 0) or 0),
            "topk_gc_total_mb": _mb(native.get("topk_gc_total_bytes", 0) or 0),
            "topk_gc_total_mb_per_query": _mb(native.get("topk_gc_total_bytes", 0) or 0) / query_count,
            "semantic_topk_gc_bytes": int(native.get("semantic_topk_gc_bytes", 0) or 0),
            "lexical_topk_gc_bytes": int(native.get("lexical_topk_gc_bytes", 0) or 0),
            "suda_pir_query_bytes": int(native.get("suda_pir_query_bytes", 0) or 0),
            "suda_pir_response_bytes": int(native.get("suda_pir_response_bytes", 0) or 0),
            "suda_pir_upload_mb": _mb(native.get("suda_pir_query_bytes", 0) or 0),
            "suda_pir_download_mb": _mb(native.get("suda_pir_response_bytes", 0) or 0),
            "suda_pir_upload_mb_per_query": _mb(native.get("suda_pir_query_bytes", 0) or 0) / query_count,
            "suda_pir_download_mb_per_query": _mb(native.get("suda_pir_response_bytes", 0) or 0) / query_count,
        },
        "paper_references": {},
    }

    references = PAPER_EFFICIENCY_REFERENCES.get(report["dataset_key"] or "", {})
    if references:
        lexical_actual = stage_directional[MULTLPSI_STAGE]
        lexical_time = _stage_seconds(timings, batch_summary, MULTLPSI_STAGE)
        lexical_ref = references["lexical_multlpsi"]
        p4_cache = data.get("p4_setup_cache") if isinstance(data.get("p4_setup_cache"), dict) else {}
        p4_cached_client_bytes = int(p4_cache.get("client_bytes", 0) or 0)
        report["paper_references"]["table5_lexical_multlpsi"] = {
            "paper": lexical_ref,
            "actual_scope": (
                "Protocol 4 / MultLPSI online socket traffic. If p4 setup cache is ready, the OKVS setup "
                "object was loaded locally and is shown separately as cached_setup_download_mb."
            ),
            "actual": {
                "time_seconds": lexical_time,
                "upload_mb": lexical_actual["upload_mb"] / query_count,
                "download_mb": lexical_actual["download_mb"] / query_count,
                "cached_setup_download_mb_once": _mb(p4_cached_client_bytes),
                "cached_setup_download_mb_amortized_per_query": _mb(p4_cached_client_bytes) / query_count,
                "download_mb_plus_cached_setup_amortized": (
                    lexical_actual["download_mb"] + _mb(p4_cached_client_bytes)
                ) / query_count,
            },
            "actual_over_paper": {
                "time": _ratio(lexical_time, lexical_ref["time_seconds"]) if lexical_time is not None else None,
                "upload": _ratio(lexical_actual["upload_mb"] / query_count, lexical_ref["upload_mb"]),
                "download": _ratio(lexical_actual["download_mb"] / query_count, lexical_ref["download_mb"]),
                "download_plus_cached_setup_amortized": _ratio(
                    (lexical_actual["download_mb"] + _mb(p4_cached_client_bytes)) / query_count,
                    lexical_ref["download_mb"],
                ),
            },
        }

        semantic_actual = stage_directional[OBLIVIOUS_FILTER_STAGE]
        semantic_time = _stage_seconds(timings, batch_summary, OBLIVIOUS_FILTER_STAGE)
        semantic_ref = references["semantic_coarse_to_fine"]
        p3_cache = data.get("p3_setup_cache") if isinstance(data.get("p3_setup_cache"), dict) else {}
        p3_cached_client_bytes = int(p3_cache.get("client_bytes", 0) or 0)
        report["paper_references"]["table4_semantic_oblivious_filter_partial"] = {
            "paper": semantic_ref,
            "actual_scope": (
                "Protocol 3 / Oblivious Filter socket traffic only. Paper Table 4 covers the full "
                "semantic coarse-to-fine retrieval path, so this is a partial lower-level comparison. "
                "If p3 setup cache is ready, the setup object is shown separately as cached_setup_download_mb."
            ),
            "actual": {
                "time_seconds": semantic_time,
                "upload_mb": semantic_actual["upload_mb"] / query_count,
                "download_mb": semantic_actual["download_mb"] / query_count,
                "cached_setup_download_mb_once": _mb(p3_cached_client_bytes),
                "cached_setup_download_mb_amortized_per_query": _mb(p3_cached_client_bytes) / query_count,
                "download_mb_plus_cached_setup_amortized": (
                    semantic_actual["download_mb"] + _mb(p3_cached_client_bytes)
                ) / query_count,
            },
            "actual_over_paper": {
                "time": _ratio(semantic_time, semantic_ref["time_seconds"]) if semantic_time is not None else None,
                "upload": _ratio(semantic_actual["upload_mb"] / query_count, semantic_ref["upload_mb"]),
                "download": _ratio(semantic_actual["download_mb"] / query_count, semantic_ref["download_mb"]),
                "download_plus_cached_setup_amortized": _ratio(
                    (semantic_actual["download_mb"] + _mb(p3_cached_client_bytes)) / query_count,
                    semantic_ref["download_mb"],
                ),
            },
        }
    return report


def bert_from_word_embeddings(model, word_embeddings, pos_oh, type_oh, attention_mask=None):
    position_embeds = model.embeddings.position_embeddings(pos_oh)
    token_type_embeds = model.embeddings.token_type_embeddings(type_oh)
    embedding_output = model.embeddings.LayerNorm(word_embeddings + position_embeds + token_type_embeds)

    extended_mask = None
    if attention_mask is not None:
        one = RingTensor.convert_to_ring(1.0).to(attention_mask.device)
        neg_val = RingTensor.convert_to_ring(-10000.0).to(attention_mask.device)
        extended_mask = (one - attention_mask) * neg_val
        extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)

    sequence_output = model.encoder(embedding_output, extended_mask)
    pooled_output = model.pooler(sequence_output)
    return sequence_output, pooled_output


def query_word_embeddings_for_joint(model, query_oh_share, joint_batch, total_seq):
    if int(query_oh_share.shape[0]) != int(joint_batch):
        query_oh_share = ArithmeticSecretSharing.cat([query_oh_share for _ in range(int(joint_batch))], dim=0)
    pad_len = int(total_seq) - int(query_oh_share.shape[1])
    if pad_len < 0:
        raise ValueError("query length cannot exceed total sequence length")
    if pad_len:
        zeros = torch.zeros(
            int(joint_batch),
            pad_len,
            BERT_CONFIG["vocab_size"],
            dtype=torch.long,
            device=DEVICE,
        )
        zero_share = ArithmeticSecretSharing(RingTensor(zeros, dtype="float", device=DEVICE))
        padded_query = ArithmeticSecretSharing.cat([query_oh_share, zero_share], dim=1)
    else:
        padded_query = query_oh_share
    return model.embeddings.word_embeddings(padded_query)[:, :QUERY_LEN]


def client_owned_topk_ids(indicators):
    local = indicators.item.convert_to_real_field().round().cpu()
    if local.ndim != 2:
        raise ValueError("top-k indicators must be a 2-D matrix")
    row_sums = local.sum(dim=1)
    if not torch.all(row_sums == 1):
        raise ValueError("client-owned top-k indicators are not one-hot on this party")
    return local.argmax(dim=1).to(dtype=torch.long)


def semantic_candidates_from_ids(source_tensor, candidate_ids):
    """Build the Protocol-1 D' tensor exactly from server-recovered candidates.

    Pisces Protocol 3 lets S learn D'.  The following Protocol 1 fine stage
    should run on D', not the full document database and not padded dummy rows.
    """

    ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=source_tensor.device).reshape(-1)
    valid_count = int(ids.numel())
    valid_mask = torch.ones(valid_count, dtype=torch.float32, device=source_tensor.device)
    out_shape = (valid_count,) + tuple(source_tensor.shape[1:])
    values = torch.zeros(out_shape, dtype=source_tensor.dtype, device=source_tensor.device)
    if valid_count:
        values = source_tensor[ids]
    return ids, valid_mask, values


def pad_ass_rows(share, target_rows):
    """Pad retrieved secret-shared documents only for the fixed-shape BERT input."""

    current_rows = int(share.shape[0])
    target_rows = int(target_rows)
    if current_rows == target_rows:
        return share
    if current_rows > target_rows:
        raise ValueError("cannot pad a share with more rows than target_rows")
    zeros = torch.zeros(
        (target_rows - current_rows,) + tuple(int(dim) for dim in share.shape[1:]),
        dtype=torch.long,
        device=DEVICE,
    )
    zero_share = ArithmeticSecretSharing(RingTensor(zeros, dtype="float", device=DEVICE))
    return ArithmeticSecretSharing.cat([share, zero_share], dim=0)


def zero_document_share(rows):
    zeros = torch.zeros(
        int(rows),
        SEM_DOC_LEN,
        BERT_CONFIG["hidden_size"],
        dtype=torch.long,
        device=DEVICE,
    )
    return ArithmeticSecretSharing(RingTensor(zeros, dtype="float", device=DEVICE))


def joint_attention_mask_for_fusion(joint_batch, semantic_rows):
    """Public BERT mask for rows where the semantic branch returned fewer than top-k docs."""

    joint_mask = torch.ones(int(joint_batch), TOTAL_SEQ, dtype=torch.float32, device=DEVICE)
    semantic_rows = int(semantic_rows)
    if semantic_rows < int(joint_batch):
        joint_mask[semantic_rows:, QUERY_LEN : QUERY_LEN + SEM_DOC_LEN] = 0.0
    return joint_mask


def map_candidate_positions_to_doc_ids(position_ids, candidate_ids):
    positions = torch.as_tensor(position_ids, dtype=torch.long).reshape(-1).cpu()
    candidates = torch.as_tensor(candidate_ids, dtype=torch.long).reshape(-1).cpu()
    mapped = []
    for position in positions.tolist():
        if 0 <= position < int(candidates.numel()):
            mapped.append(int(candidates[position].item()))
        else:
            mapped.append(-1)
    return mapped


def pad_doc_ids_for_fixed_pir(doc_ids, target_rows, database_size):
    real_ids = [int(doc_id) for doc_id in doc_ids if 0 <= int(doc_id) < int(database_size)]
    if len(real_ids) > int(target_rows):
        raise ValueError("cannot pad more document ids than target_rows")
    used = set(real_ids)
    padded = list(real_ids)
    for candidate in range(int(database_size)):
        if len(padded) >= int(target_rows):
            break
        if candidate not in used:
            padded.append(candidate)
            used.add(candidate)
    if len(padded) != int(target_rows):
        raise ValueError("not enough document ids to pad fixed-size PIR query")
    return torch.tensor(padded, dtype=torch.long, device=DEVICE)


def indicator_from_candidate_doc_ids(position_ids, candidate_ids, top_k, num_docs):
    doc_ids = [
        doc_id
        for doc_id in map_candidate_positions_to_doc_ids(position_ids, candidate_ids)
        if doc_id >= 0
    ]
    if len(doc_ids) < int(top_k):
        doc_ids.extend([-1] * (int(top_k) - len(doc_ids)))
    indicators = torch.zeros(int(top_k), int(num_docs), dtype=torch.float32, device=DEVICE)
    for row, doc_id in enumerate(doc_ids[: int(top_k)]):
        if 0 <= doc_id < int(num_docs):
            indicators[row, doc_id] = 1.0
    return indicators


def _aux_param_count(param_cls, party_id=0, saved_name=None):
    base_name = saved_name or param_cls.__name__
    path = Path(param_path) / param_cls.__name__ / f"{base_name}_{party_id}.pth"
    if not path.exists():
        return 0
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    count = _first_param_axis0(payload)
    if count is None:
        raise RuntimeError(f"could not infer auxiliary parameter count from {path}")
    return int(count)


def _first_param_axis0(value):
    if hasattr(value, "shape") and len(value.shape) > 0:
        return int(value.shape[0])
    tensor = getattr(value, "tensor", None)
    if hasattr(tensor, "shape") and len(tensor.shape) > 0:
        return int(tensor.shape[0])
    item = getattr(value, "item", None)
    if item is not None:
        count = _first_param_axis0(item)
        if count is not None:
            return count
    if isinstance(value, dict):
        for child in value.values():
            count = _first_param_axis0(child)
            if count is not None:
                return count
    if isinstance(value, (list, tuple)):
        for child in value:
            count = _first_param_axis0(child)
            if count is not None:
                return count
    return None


def _ceil_with_safety(count):
    return max(1, int(np.ceil(float(count) * float(PARAM_SAFETY_FACTOR))))


def protocol2_aux_requirements(num_docs, query_terms):
    denominator_elems = int(num_docs) * int(query_terms)
    return {
        "DivKey": denominator_elems,
        "B2AKey": 2 * int(SCALE_BIT) * denominator_elems,
    }


def check_protocol2_aux_params(role, num_docs, query_terms):
    if not CHECK_PARAMS:
        debug(role, "Aux-check", "PISCES_RAG_CHECK_PARAMS=0, skip Protocol2 auxiliary parameter check")
        return protocol2_aux_requirements(num_docs, query_terms)
    requirements = protocol2_aux_requirements(num_docs, query_terms)
    available_b2a = _aux_param_count(B2AKey, party_id=0, saved_name=B2A_KEY_SAVED_NAME)
    available_div = _aux_param_count(DivKey, party_id=0, saved_name=DIV_KEY_SAVED_NAME)
    log(
        role,
        "Aux-check",
        f"Protocol2 needs DivKey>={requirements['DivKey']}, B2AKey>={requirements['B2AKey']} "
        f"(available DivKey={available_div}, B2AKey={available_b2a})",
    )
    missing = []
    if available_div < requirements["DivKey"]:
        missing.append(f"DivKey {available_div}<{requirements['DivKey']}")
    if available_b2a < requirements["B2AKey"]:
        missing.append(f"B2AKey {available_b2a}<{requirements['B2AKey']}")
    if missing:
        raise RuntimeError(
            "Protocol 2 auxiliary parameters are insufficient: "
            + ", ".join(missing)
            + ". Generate more parameters first, e.g. set PISCES_RAG_B2A_KEY_COUNT="
            + str(max(B2A_KEY_COUNT, requirements["B2AKey"]))
            + " and run rag.py without SKIP_GEN_PARAMS=1, or regenerate B2AKey directly."
        )
    return requirements


def check_model_aux_params(role):
    if not CHECK_PARAMS:
        required_gelu = int(TOP_K) * int(TOTAL_SEQ) * int(BERT_CONFIG["intermediate_size"])
        debug(role, "Aux-check", "PISCES_RAG_CHECK_PARAMS=0, skip Secure-BERT auxiliary parameter check")
        return {"GeLUKey": required_gelu, "SigmaDICFKey": required_gelu}
    # GeLU is called on one transformer feed-forward tensor at a time. The
    # joint RAG pass has batch=TOP_K and sequence=TOTAL_SEQ, so this is the
    # largest single GeLU-related request made by the current demo model.
    required_gelu = int(TOP_K) * int(TOTAL_SEQ) * int(BERT_CONFIG["intermediate_size"])
    available_gelu = _aux_param_count(GeLUKey, party_id=0)
    available_sigma_dicf = _aux_param_count(SigmaDICFKey, party_id=0)
    log(
        role,
        "Aux-check",
        f"Secure-BERT needs GeLUKey>={required_gelu} and SigmaDICFKey>={required_gelu} "
        f"for one joint FFN activation (available GeLUKey={available_gelu}, "
        f"SigmaDICFKey={available_sigma_dicf})",
    )
    missing = []
    if available_gelu < required_gelu:
        missing.append(f"GeLUKey {available_gelu}<{required_gelu}")
    if available_sigma_dicf < required_gelu:
        missing.append(f"SigmaDICFKey {available_sigma_dicf}<{required_gelu}")
    if missing:
        raise RuntimeError(
            "Secure-BERT auxiliary parameters are insufficient: "
            + ", ".join(missing)
            + ". "
            "Generate more parameters first, e.g. run rag.py without SKIP_GEN_PARAMS=1 "
            f"and set PISCES_RAG_GELU_KEY_COUNT={max(GELU_KEY_COUNT, required_gelu)} "
            f"PISCES_RAG_SIGMA_DICF_KEY_COUNT={max(SIGMA_DICF_KEY_COUNT, required_gelu)}."
        )
    return {"GeLUKey": required_gelu, "SigmaDICFKey": required_gelu}


def check_native_runtime_paths(role):
    if not PANTHER_GC_TOPK_BIN.exists():
        raise RuntimeError(
            "OpenPanther GC top-k binary is required for the Pisces RAG main path, "
            f"but it was not found at {PANTHER_GC_TOPK_BIN}. "
            "Build NssMPClib/native/panther_gc_topk/panther_gc_topk_cli.cc inside OpenPanther, "
            "or set PANTHER_GC_TOPK_BIN to the compiled binary."
        )
    try:
        import importlib

        importlib.import_module("NssMPC.application.rag.pisces._suda_bridge")
    except ImportError as exc:
        raise RuntimeError(
            "Native Suda bridge is required for the Pisces RAG main path. "
            "Build the bridge and make its shared libraries visible, e.g. libntl.so.44 via LD_LIBRARY_PATH."
        ) from exc
    debug(role, "Preflight", f"OpenPanther GC top-k binary={PANTHER_GC_TOPK_BIN}")


def estimate_query_term_count():
    if REAL_DATASET_MODE:
        try:
            return int(load_real_rag_inputs()["query_bm25_tokens"].numel())
        except Exception as exc:
            default_count = REAL_QUERY_TERM_LIMIT or int(QUERY_BM25_TOKENS.numel())
            log(
                "Init",
                "Params",
                f"could not inspect real query terms before parameter check ({exc}); "
                f"default query_terms={default_count}",
            )
            return int(default_count)
    return int(QUERY_BM25_TOKENS.numel())


def current_aux_requirements():
    query_terms = estimate_query_term_count()
    p2 = protocol2_aux_requirements(NUM_DOCS, query_terms)
    p2_multiplier = max(1, int(REAL_QUERY_COUNT))
    model = {
        "GeLUKey": int(TOP_K) * int(TOTAL_SEQ) * int(BERT_CONFIG["intermediate_size"]),
        "SigmaDICFKey": int(TOP_K) * int(TOTAL_SEQ) * int(BERT_CONFIG["intermediate_size"]),
    }
    return {
        "query_terms": int(query_terms),
        "requirements": {
            "DivKey": int(p2["DivKey"]) * p2_multiplier,
            "B2AKey": int(p2["B2AKey"]) * p2_multiplier,
            "GeLUKey": int(model["GeLUKey"]),
            "SigmaDICFKey": int(model["SigmaDICFKey"]),
        },
    }


def ensure_aux_param(param_cls, required, *, configured_count, saved_name=None):
    required = int(required)
    available = _aux_param_count(param_cls, party_id=0, saved_name=saved_name)
    target = max(int(configured_count), _ceil_with_safety(required))
    name = saved_name or param_cls.__name__
    if available >= required:
        log("Init", "Params", f"{name}: available={available}, required={required}, ok")
        return {"available": available, "required": required, "generated": False, "target": available}
    if not AUTO_GEN_PARAMS:
        raise RuntimeError(
            f"{name} auxiliary parameters are insufficient: available={available}, required={required}. "
            "Set PISCES_RAG_AUTO_GEN_PARAMS=1 or generate parameters manually."
        )
    log(
        "Init",
        "Params",
        f"{name}: available={available}, required={required}, regenerate target={target}",
    )
    if saved_name is None:
        param_cls.gen_and_save(target)
    else:
        param_cls.gen_and_save(target, saved_name=saved_name)
    regenerated = _aux_param_count(param_cls, party_id=0, saved_name=saved_name)
    if regenerated < required:
        raise RuntimeError(
            f"{name} regeneration finished but is still insufficient: "
            f"available={regenerated}, required={required}"
        )
    return {"available": regenerated, "required": required, "generated": True, "target": target}


def ensure_aux_params_for_current_run():
    if not CHECK_PARAMS:
        log("Init", "Params", "PISCES_RAG_CHECK_PARAMS=0, skip auxiliary parameter check")
        update_metrics(
            "aux_params",
            {
                "check_enabled": False,
                "auto_gen_enabled": bool(AUTO_GEN_PARAMS),
            },
        )
        return
    estimate = current_aux_requirements()
    requirements = estimate["requirements"]
    log(
        "Init",
        "Params",
        f"auto-check enabled={AUTO_GEN_PARAMS}, docs={NUM_DOCS}, top_k={TOP_K}, "
        f"total_seq={TOTAL_SEQ}, query_terms={estimate['query_terms']}, query_count={REAL_QUERY_COUNT}, "
        f"safety_factor={PARAM_SAFETY_FACTOR}",
    )
    results = {
        "DivKey": ensure_aux_param(
            DivKey,
            requirements["DivKey"],
            configured_count=DIV_KEY_COUNT,
            saved_name=DIV_KEY_SAVED_NAME,
        ),
        "B2AKey": ensure_aux_param(
            B2AKey,
            requirements["B2AKey"],
            configured_count=B2A_KEY_COUNT,
            saved_name=B2A_KEY_SAVED_NAME,
        ),
        "GeLUKey": ensure_aux_param(GeLUKey, requirements["GeLUKey"], configured_count=GELU_KEY_COUNT),
        "SigmaDICFKey": ensure_aux_param(
            SigmaDICFKey,
            requirements["SigmaDICFKey"],
            configured_count=SIGMA_DICF_KEY_COUNT,
        ),
    }
    update_metrics(
        "aux_params",
        {
            "auto_gen_enabled": bool(AUTO_GEN_PARAMS),
            "check_enabled": True,
            "safety_factor": float(PARAM_SAFETY_FACTOR),
            "query_terms": int(estimate["query_terms"]),
            "requirements": requirements,
            "results": results,
        },
    )
    generated = [name for name, result in results.items() if result["generated"]]
    if generated:
        log("Init", "Params", f"regenerated: {generated}")
    else:
        log("Init", "Params", "all checked parameters are sufficient")


def _stable_json_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _setup_cache_paths(kind, cache_key):
    root = SETUP_CACHE_DIR / cache_key
    return {
        "server": root / f"{kind}_server.pkl",
        "client": root / f"{kind}_client.pkl",
    }


def _setup_cache_ready(kind, cache_key):
    if not SETUP_CACHE_ENABLED:
        return False
    paths = _setup_cache_paths(kind, cache_key)
    return paths["server"].exists() and paths["client"].exists()


def _setup_cache_info(kind, cache_key):
    paths = _setup_cache_paths(kind, cache_key)
    info = {
        "kind": kind,
        "cache_key": cache_key,
        "ready": _setup_cache_ready(kind, cache_key),
    }
    for role, path in paths.items():
        info[f"{role}_path"] = str(path)
        info[f"{role}_bytes"] = int(path.stat().st_size) if path.exists() else 0
    return info


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


REAL_CONTEXT = None
REAL_INPUTS_BY_QUERY = {}


def _path_fingerprint(path):
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_real_rag_context():
    global REAL_CONTEXT
    if REAL_CONTEXT is not None:
        return REAL_CONTEXT
    with _REAL_CONTEXT_LOCK:
        if REAL_CONTEXT is not None:
            return REAL_CONTEXT
        return _load_real_rag_context_unlocked()


def _load_real_rag_context_unlocked():
    global REAL_CONTEXT
    plaintext_root = Path("/home/adslab/pazika/pisces")
    sys.path.append(str(plaintext_root))
    sys.path.append(str(plaintext_root / "scripts"))
    from verify_datasets import prepare_hotpotqa, prepare_squad
    from transformers import AutoTokenizer

    if REAL_DATASET_NAME == "squad_dev_v2":
        prepared = prepare_squad("squad_dev_v2", REAL_SQUAD_PATH)
        chunk_embeddings_path = (
            REAL_CACHE_DIR / "squad_dev_v2_chunks_ibm-granite_granite-embedding-small-english-r2_cls_max8192_1204.npy"
        )
        query_embeddings_path = (
            REAL_CACHE_DIR / "squad_dev_v2_queries_5928_ibm-granite_granite-embedding-small-english-r2_cls_max8192_5928.npy"
        )
        term_frequencies_path = REAL_CACHE_DIR / "squad_dev_v2_bert_terms_bert-base-uncased_1204.pkl"
    elif REAL_DATASET_NAME == "hotpotqa_dev_distractor":
        prepared = prepare_hotpotqa("hotpotqa_dev_distractor", REAL_HOTPOT_PATH)
        chunk_embeddings_path = (
            REAL_CACHE_DIR
            / "hotpotqa_dev_distractor_chunks_ibm-granite_granite-embedding-small-english-r2_cls_max8192_269602.npy"
        )
        query_embeddings_path = (
            REAL_CACHE_DIR
            / "hotpotqa_dev_distractor_queries_7405_ibm-granite_granite-embedding-small-english-r2_cls_max8192_7405.npy"
        )
        term_frequencies_path = REAL_CACHE_DIR / "hotpotqa_dev_distractor_bert_terms_bert-base-uncased_269602.pkl"
    else:
        raise ValueError(
            "rag.py real-data mode supports PISCES_RAG_DATASET=squad_dev_v2 "
            "or hotpotqa_dev_distractor"
        )

    chunk_embeddings = np.load(chunk_embeddings_path, mmap_mode="r")
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    with term_frequencies_path.open("rb") as handle:
        term_frequencies = pickle.load(handle)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    corpus_terms = term_frequencies[:NUM_DOCS]
    vocabulary = {}
    for counter in corpus_terms:
        for term in counter:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)

    def query_term_info(text):
        with _TOKENIZER_LOCK:
            raw_terms = tokenizer.tokenize(text)
        hit_terms = [term for term in raw_terms if term in vocabulary]
        if REAL_QUERY_TERM_LIMIT > 0:
            selected_terms = hit_terms[:REAL_QUERY_TERM_LIMIT]
        else:
            selected_terms = hit_terms
        return raw_terms, hit_terms, selected_terms

    doc_token_rows = []
    for chunk in prepared.chunks[:NUM_DOCS]:
        with _TOKENIZER_LOCK:
            encoded = tokenizer(
                chunk.text,
                add_special_tokens=True,
                truncation=True,
                max_length=SEM_DOC_LEN,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].reshape(-1)
        doc_token_rows.append(encoded)
    db_tokens_ids = torch.stack(doc_token_rows, dim=0).to(DEVICE)

    document_tf_plain = torch.zeros(max(1, len(vocabulary)), NUM_DOCS, dtype=torch.float32, device=DEVICE)
    document_tf_entries = []
    for doc_id, counter in enumerate(corpus_terms):
        for term, frequency in counter.items():
            token_id = vocabulary.get(term)
            if token_id is not None:
                document_tf_plain[token_id, doc_id] = float(frequency)
                if frequency > 0:
                    document_tf_entries.append((int(token_id), int(doc_id), int(frequency)))

    p3_setup_cache_payload = {
        "dataset": REAL_DATASET_NAME,
        "num_docs": int(NUM_DOCS),
        "embedding_cache": _path_fingerprint(chunk_embeddings_path),
        "p3": {
            "simhash_bits": int(P3_SIMHASH_BITS),
            "threshold": int(P3_THRESHOLD),
            "projection_count": int(P3_PROJECTION_COUNT),
            "paillier_key_size": int(P3_PAILLIER_KEY_SIZE),
            "okvs_expansion": 3.0,
            "encoding": "single-ciphertext-per-projection",
        },
    }
    p4_setup_cache_payload = {
        "dataset": REAL_DATASET_NAME,
        "num_docs": int(NUM_DOCS),
        "term_cache": _path_fingerprint(term_frequencies_path),
        "vocab_size": int(len(vocabulary)),
        "tf_entries": int(len(document_tf_entries)),
        "p4": {
            "okvs_expansion": float(P4_OKVS_EXPANSION),
            "oprf_secret_key": int(P4_OPRF_SECRET_KEY),
            "prefix_size": 16,
            "tf_size": 8,
        },
    }
    p3_setup_cache_key = _stable_json_digest(p3_setup_cache_payload)
    p4_setup_cache_key = _stable_json_digest(p4_setup_cache_payload)

    REAL_CONTEXT = {
        "prepared": prepared,
        "tokenizer": tokenizer,
        "query_embeddings": query_embeddings,
        "vocabulary": vocabulary,
        "query_term_info": query_term_info,
        "db_embeddings": torch.tensor(np.asarray(chunk_embeddings[:NUM_DOCS]), dtype=torch.float32, device=DEVICE),
        "db_tokens_ids": db_tokens_ids,
        "document_tf_plain": document_tf_plain,
        "document_tf_entries": tuple(document_tf_entries),
        "p3_setup_cache_key": p3_setup_cache_key,
        "p4_setup_cache_key": p4_setup_cache_key,
        "p3_setup_cache_ready": _setup_cache_ready("p3", p3_setup_cache_key),
        "p4_setup_cache_ready": _setup_cache_ready("p4", p4_setup_cache_key),
        "p3_setup_cache_info": _setup_cache_info("p3", p3_setup_cache_key),
        "p4_setup_cache_info": _setup_cache_info("p4", p4_setup_cache_key),
        "bm25_vocab": int(len(vocabulary)),
    }
    return REAL_CONTEXT


def _real_query_is_usable(gold_ids, hit_terms):
    if not all(gold_id < NUM_DOCS for gold_id in gold_ids):
        return False
    if REAL_REQUIRE_QUERY_TERMS and REAL_QUERY_TERM_LIMIT > 0 and len(hit_terms) < REAL_QUERY_TERM_LIMIT:
        return False
    return True


def get_real_query_indices(count=None, start_index=None):
    context = load_real_rag_context()
    prepared = context["prepared"]
    query_term_info = context["query_term_info"]
    target_count = int(count if count is not None else REAL_QUERY_COUNT)
    index = int(start_index if start_index is not None else REAL_QUERY_INDEX)
    skipped_usable = 0
    selected = []
    while index < len(prepared.queries) and len(selected) < target_count:
        query_text, gold_ids = prepared.queries[index]
        _, hit_terms, _ = query_term_info(query_text)
        if _real_query_is_usable(gold_ids, hit_terms):
            if start_index is None and skipped_usable < REAL_QUERY_OFFSET:
                skipped_usable += 1
            else:
                selected.append(index)
        index += 1
    if len(selected) < target_count:
        raise ValueError(
            f"only found {len(selected)} usable {REAL_DATASET_NAME} queries from index "
            f"{start_index if start_index is not None else REAL_QUERY_INDEX}; requested {target_count}"
        )
    return selected


def load_real_rag_inputs(query_index=None):
    global REAL_INPUTS
    requested_index = int(REAL_QUERY_INDEX if query_index is None else query_index)
    if query_index is None and REAL_INPUTS is not None and int(REAL_INPUTS["requested_query_index"]) == requested_index:
        return REAL_INPUTS
    if requested_index in REAL_INPUTS_BY_QUERY:
        inputs = REAL_INPUTS_BY_QUERY[requested_index]
        if query_index is None:
            REAL_INPUTS = inputs
        return inputs

    context = load_real_rag_context()
    prepared = context["prepared"]
    tokenizer = context["tokenizer"]
    query_term_info = context["query_term_info"]

    query_index_actual = requested_index
    query_text, gold_ids = prepared.queries[query_index_actual]
    raw_query_terms, hit_query_terms, selected_query_terms = query_term_info(query_text)
    if not _real_query_is_usable(gold_ids, hit_query_terms):
        query_index_actual = get_real_query_indices(1, start_index=requested_index)[0]
        query_text, gold_ids = prepared.queries[query_index_actual]
        raw_query_terms, hit_query_terms, selected_query_terms = query_term_info(query_text)

    with _TOKENIZER_LOCK:
        query_ids = tokenizer(
            query_text,
            add_special_tokens=True,
            truncation=True,
            max_length=QUERY_LEN,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(dtype=torch.long)

    query_bm25_tokens = torch.tensor(
        [context["vocabulary"][term] for term in selected_query_terms],
        dtype=torch.long,
        device=DEVICE,
    )
    if query_bm25_tokens.numel() == 0:
        query_bm25_tokens = torch.zeros(0, dtype=torch.long, device=DEVICE)

    inputs = {
        "requested_query_index": requested_index,
        "query_index": query_index_actual,
        "query_text": query_text,
        "gold_ids": sorted(gold_ids),
        "db_embeddings": context["db_embeddings"],
        "semantic_query_embedding": torch.tensor(
            np.asarray(context["query_embeddings"][query_index_actual]),
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0),
        "query_ids": query_ids.to(DEVICE),
        "db_tokens_ids": context["db_tokens_ids"],
        "document_tf_plain": context["document_tf_plain"],
        "document_tf_entries": context["document_tf_entries"],
        "query_bm25_tokens": query_bm25_tokens,
        "query_terms": raw_query_terms,
        "query_bm25_hit_terms": hit_query_terms,
        "query_bm25_selected_terms": selected_query_terms,
        "p3_setup_cache_key": context["p3_setup_cache_key"],
        "p4_setup_cache_key": context["p4_setup_cache_key"],
        "p3_setup_cache_ready": context["p3_setup_cache_ready"],
        "p4_setup_cache_ready": context["p4_setup_cache_ready"],
        "p3_setup_cache_info": context["p3_setup_cache_info"],
        "p4_setup_cache_info": context["p4_setup_cache_info"],
    }
    REAL_INPUTS_BY_QUERY[requested_index] = inputs
    if query_index is None:
        REAL_INPUTS = inputs
    update_metrics(
        "data",
        {
            "dataset": REAL_DATASET_NAME,
            "query_index": int(query_index_actual),
            "query": query_text,
            "gold_ids": sorted(int(value) for value in gold_ids),
            "num_docs": int(NUM_DOCS),
            "embedding_dim": int(inputs["db_embeddings"].shape[-1]),
            "bm25_vocab": int(context["bm25_vocab"]),
            "raw_query_terms": list(raw_query_terms),
            "query_bm25_hit_terms": list(hit_query_terms),
            "query_bm25_selected_terms": list(selected_query_terms),
            "query_bm25_hits": int(query_bm25_tokens.numel()),
            "query_term_limit": int(REAL_QUERY_TERM_LIMIT),
            "require_query_terms": bool(REAL_REQUIRE_QUERY_TERMS),
            "p3_setup_cache": context["p3_setup_cache_info"],
            "p4_setup_cache": context["p4_setup_cache_info"],
        },
    )
    log(
        "Data",
        "Load",
        f"dataset={REAL_DATASET_NAME}, query_index={query_index_actual}, gold_ids={sorted(gold_ids)}, "
        f"query={query_text!r}, docs={NUM_DOCS}, embedding_dim={inputs['db_embeddings'].shape[-1]}, "
        f"bm25_vocab={context['bm25_vocab']}, query_bm25_hits={query_bm25_tokens.numel()}, "
        f"selected_terms={selected_query_terms}",
    )
    return inputs


def demo_semantic_query_embedding():
    if REAL_DATASET_MODE:
        return load_real_rag_inputs()["semantic_query_embedding"]
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260624)
    return torch.randn(1, BERT_CONFIG["hidden_size"], generator=generator, device=DEVICE)


def gen_params():
    log("Init", "Params", "generate auxiliary parameters")
    if not os.path.exists('data'): os.makedirs('data')
    AssMulTriples.gen_and_save(ASS_MUL_TRIPLE_COUNT, saved_name='2PCBeaver')
    DivKey.gen_and_save(DIV_KEY_COUNT)
    GeLUKey.gen_and_save(GELU_KEY_COUNT)
    TanhKey.gen_and_save(TANH_KEY_COUNT)
    #MatmulTriples.gen_and_save(10000)
    Wrap.gen_and_save(WRAP_KEY_COUNT)
    ReciprocalSqrtKey.gen_and_save(RECIPROCAL_SQRT_KEY_COUNT)
    SigmaDICFKey.gen_and_save(SIGMA_DICF_KEY_COUNT)
    B2AKey.gen_and_save(B2A_KEY_COUNT)
    log(
        "Init",
        "Params",
        f"counts: AssMulTriples={ASS_MUL_TRIPLE_COUNT}, DivKey={DIV_KEY_COUNT}, "
        f"GeLUKey={GELU_KEY_COUNT}, TanhKey={TANH_KEY_COUNT}, Wrap={WRAP_KEY_COUNT}, "
        f"ReciprocalSqrtKey={RECIPROCAL_SQRT_KEY_COUNT}, SigmaDICFKey={SIGMA_DICF_KEY_COUNT}, "
        f"B2AKey={B2A_KEY_COUNT}",
    )
    log("Init", "Params", "done")


def write_report_if_requested():
    if not REAL_REPORT_JSON:
        return
    report_path = Path(REAL_REPORT_JSON)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    update_metrics(
        "config",
        {
            "profile": RAG_PROFILE or "custom",
            "real_dataset": bool(REAL_DATASET_MODE),
            "dataset": REAL_DATASET_NAME,
            "num_docs": int(NUM_DOCS),
            "query_count": int(REAL_QUERY_COUNT),
            "query_offset": int(REAL_QUERY_OFFSET),
            "top_k": int(TOP_K),
            "doc_len": int(SEM_DOC_LEN),
            "query_len": int(QUERY_LEN),
            "query_term_limit": int(REAL_QUERY_TERM_LIMIT),
            "require_query_terms": bool(REAL_REQUIRE_QUERY_TERMS),
            "p3_paillier_key_size": int(P3_PAILLIER_KEY_SIZE),
            "p3_simhash_bits": int(P3_SIMHASH_BITS),
            "p3_threshold": int(P3_THRESHOLD),
            "p3_projections": int(P3_PROJECTION_COUNT),
            "p4_okvs_expansion": float(P4_OKVS_EXPANSION),
            "payload_pir_scope": "full-db",
            "setup_cache_enabled": bool(SETUP_CACHE_ENABLED),
            "bert_weights_path": str(BERT_WEIGHTS_PATH),
            "load_bert_weights": bool(LOAD_BERT_WEIGHTS),
        },
    )
    retrieval_stages = {
        OBLIVIOUS_FILTER_STAGE,
        MULTLPSI_STAGE,
        "Protocol2-bm25",
        "TopK",
        "Suda-PIR",
    }
    communication = RUN_METRICS.get("communication", {})
    batch_summary = RUN_METRICS.get("batch_summary", {})
    if isinstance(batch_summary, dict) and batch_summary.get("stage_communication"):
        communication = batch_summary["stage_communication"]
    summary = {}
    role_totals = {}
    for role in ("Server", "Client"):
        role_items = {
            key: value
            for key, value in communication.items()
            if key.startswith(f"{role}.")
        }
        total_send = sum(int(value.get("send_bytes", 0)) for value in role_items.values())
        total_recv = sum(int(value.get("recv_bytes", 0)) for value in role_items.values())
        retrieval_send = 0
        retrieval_recv = 0
        for key, value in role_items.items():
            stage = key.split(".", 1)[1]
            if stage in retrieval_stages:
                retrieval_send += int(value.get("send_bytes", 0))
                retrieval_recv += int(value.get("recv_bytes", 0))
        role_totals[role] = {
            "total_send": total_send,
            "total_recv": total_recv,
            "retrieval_send": retrieval_send,
            "retrieval_recv": retrieval_recv,
        }
        summary[f"{role}.total_send_bytes"] = total_send
        summary[f"{role}.total_recv_bytes"] = total_recv
        summary[f"{role}.retrieval_send_bytes"] = retrieval_send
        summary[f"{role}.retrieval_recv_bytes"] = retrieval_recv
        summary[f"{role}.model_and_setup_send_bytes"] = total_send - retrieval_send
        summary[f"{role}.model_and_setup_recv_bytes"] = total_recv - retrieval_recv
    if {"Server", "Client"}.issubset(role_totals):
        # One direction is observed as sender bytes by one party and receiver
        # bytes by the other; use max to avoid double-counting the same link.
        server_to_client = max(role_totals["Server"]["total_send"], role_totals["Client"]["total_recv"])
        client_to_server = max(role_totals["Client"]["total_send"], role_totals["Server"]["total_recv"])
        retrieval_server_to_client = max(
            role_totals["Server"]["retrieval_send"],
            role_totals["Client"]["retrieval_recv"],
        )
        retrieval_client_to_server = max(
            role_totals["Client"]["retrieval_send"],
            role_totals["Server"]["retrieval_recv"],
        )
        summary["unique.server_to_client_bytes"] = server_to_client
        summary["unique.client_to_server_bytes"] = client_to_server
        summary["unique.total_directional_bytes"] = server_to_client + client_to_server
        summary["unique.retrieval_server_to_client_bytes"] = retrieval_server_to_client
        summary["unique.retrieval_client_to_server_bytes"] = retrieval_client_to_server
        summary["unique.retrieval_total_directional_bytes"] = (
            retrieval_server_to_client + retrieval_client_to_server
        )
        summary["unique.model_and_setup_total_directional_bytes"] = (
            server_to_client + client_to_server
            - retrieval_server_to_client
            - retrieval_client_to_server
        )
    update_metrics("summary", summary)
    update_metrics("paper_comparison", build_paper_comparison_report())
    payload = dict(RUN_METRICS)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    log("Report", "JSON", f"wrote {report_path}")

# ==========================================
# 2. Server 与 Client 线程逻辑
# ==========================================
server = NeuralNetworkCS(type='server')
client = NeuralNetworkCS(type='client')

server.set_comparison_provider()
client.set_comparison_provider()

for p in [server, client]:
    p.set_multiplication_provider()
    p.set_comparison_provider()
    p.set_nonlinear_operation_provider()


def apply_aux_param_saved_names(party):
    if DivKey.__name__ in party.providers:
        party.providers[DivKey.__name__].saved_name = DIV_KEY_SAVED_NAME
    if B2AKey.__name__ in party.providers:
        party.providers[B2AKey.__name__].saved_name = B2A_KEY_SAVED_NAME


for p in [server, client]:
    apply_aux_param_saved_names(p)

# Single-query end-to-end protocol path, server side.
def run_server():
    run_start = time.perf_counter()
    checkpoint = CheckpointReporter("Server", run_start)
    server.online()
    checkpoint.mark("Connect")
    with PartyRuntime(server):
        print_pisces_contract("Server")
        check_model_aux_params("Server")
        check_native_runtime_paths("Server")
        checkpoint.mark("Preflight")
        # ---------------------------------------------------------
        # 1. 准备模型 (Encoder)
        # ---------------------------------------------------------
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        model = load_bert_tiny_weights(model, "Server")
        model_for_dummy = SecBertModel(BERT_CONFIG)
        for param in model_for_dummy.parameters():
            param.requires_grad = False
        model_for_dummy = load_bert_tiny_weights(model_for_dummy, "Server")

        checkpoint.mark("Build-model")
        server.dummy_model(model_for_dummy)
        word_embedding_table_for_pir = model.embeddings.word_embeddings.weight.detach().cpu().clone()
        s_local, s_remote = share_model(model)
        server.send(s_remote)
        model = load_model(model, s_local)
        
        checkpoint.mark("Setup-model")
        # ---------------------------------------------------------
        # 2. 准备服务端知识库 (Documents Database)
        # ---------------------------------------------------------
        real_inputs = load_real_rag_inputs() if REAL_DATASET_MODE else None
        if real_inputs is None:
            db_generator = torch.Generator(device=DEVICE)
            db_generator.manual_seed(20260625)
            db_embeddings = torch.randn(
                NUM_DOCS,
                BERT_CONFIG['hidden_size'],
                generator=db_generator,
                device=DEVICE,
            )
            db_embeddings[0] = demo_semantic_query_embedding()[0]
        else:
            db_embeddings = real_inputs["db_embeddings"]
        
        s_db_local, s_db_remote = share_data(db_embeddings)
        server.send(s_db_remote)
        my_db_share = s_db_local[0] 
        audit_ass_pair(
            "Server",
            server,
            "Setup-database",
            "Dv.document_embeddings",
            my_db_share,
            expected=db_embeddings,
            plaintext_visibility="S input; secret-shared before retrieval",
        )

        if real_inputs is None:
            document_tf_plain = torch.randint(
                0,
                4,
                (VOCAB_SIZE_BM25, NUM_DOCS),
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            document_tf_plain = real_inputs["document_tf_plain"]
        document_lengths_plain = document_tf_plain.sum(dim=0).clamp_min(1.0)
        query_bm25_tokens_for_debug = real_inputs["query_bm25_tokens"] if real_inputs is not None else QUERY_BM25_TOKENS
        check_protocol2_aux_params("Server", NUM_DOCS, int(query_bm25_tokens_for_debug.numel()))
        debug("Server", "P2", f"document_lengths={document_lengths_plain.tolist()}")
        debug("Server", "P2", f"average_document_length={default_average_length(document_lengths_plain):.6f}")
        for token in query_bm25_tokens_for_debug.tolist():
            debug("Server", "P4", f"token={token} plaintext_tf={document_tf_plain[token].tolist()}")

        if real_inputs is None:
            db_tokens_ids = torch.randint(
                0,
                BERT_CONFIG['vocab_size'],
                (NUM_DOCS, SEM_DOC_LEN),
                device=DEVICE,
            )
        else:
            db_tokens_ids = real_inputs["db_tokens_ids"]
        db_embedding_payload = torch.round(
            word_embedding_table_for_pir[db_tokens_ids.detach().cpu().long()] * int(float_scale)
        ).to(dtype=torch.float32)
        log(
            "Server",
            "Setup-database",
            f"db_embeddings={tuple(db_embeddings.shape)}, bm25_tf={tuple(document_tf_plain.shape)}, "
            f"avg_doc_len={default_average_length(document_lengths_plain):.4f}, "
            f"pir_payload={tuple(db_embedding_payload.shape)}, fixed_point_scale={int(float_scale)}",
        )

        checkpoint.mark("Setup-database")
        # ---------------------------------------------------------
        # 3. 接收 Client Query 并提取特征
        # ---------------------------------------------------------
        sh_in = server.receive()
        sh_pos = server.receive()
        sh_type = server.receive()
        mask = server.receive()
        audit_ass_pair("Server", server, "Query-encode", "Q.token_one_hot", sh_in[0])
        audit_ass_pair("Server", server, "Query-encode", "Q.position_one_hot", sh_pos[0])
        audit_ass_pair("Server", server, "Query-encode", "Q.type_one_hot", sh_type[0])
        _, pool = model(sh_in[0], sh_pos[0], sh_type[0], mask)
        bert_query_emb_share = pool # shape: [1, 128]
        semantic_query_remote = server.receive()
        query_emb_share = semantic_query_remote[0]
        audit_ass_pair(
            "Server",
            server,
            "Query-encode",
            "q.semantic_embedding",
            query_emb_share,
            expected=demo_semantic_query_embedding(),
        )
        debug("Server", "P3", "received semantic query embedding share")

        checkpoint.mark("Query-encode")
        #query_emb_share = server.receive()[0]
        
        # ---------------------------------------------------------
        # 4. RAG 核心流程：双路召回 (Dual-Path Retrieval)
        # ---------------------------------------------------------
        
        # Protocol 1 / ∏PrivateSS Line 1 invokes Protocol 3 / ∏Oblivious Filter.
        p3_cache_hit = False
        p3_cache_paths = None
        if real_inputs is not None and real_inputs.get("p3_setup_cache_ready"):
            p3_cache_paths = _setup_cache_paths("p3", real_inputs["p3_setup_cache_key"])
            p1_server = _load_pickle(p3_cache_paths["server"])
            p3_setup = p1_server.setup
            if p3_setup is None:
                raise RuntimeError("cached Protocol 3 server has no setup")
            p3_cache_hit = True
        else:
            p1_server = Protocol1Server.paper_defaults(
                config=RAG_CONFIG,
                paillier_key_size=P3_PAILLIER_KEY_SIZE,
                threshold=P3_THRESHOLD,
                projection_count=P3_PROJECTION_COUNT,
                simhash_bits=P3_SIMHASH_BITS,
            )
            p3_setup = p1_server.build_filter_setup(db_embeddings.cpu(), chunks=list(range(NUM_DOCS)))
        if PROTOCOL_AUDIT:
            audit_log(
                "Server",
                OBLIVIOUS_FILTER_STAGE,
                f"setup type={type(p3_setup).__name__}, public_key_bits={p3_setup.public_key[0].bit_length()}, "
                f"masks={len(p3_setup.masks)}, "
                f"okvs_slots={p3_setup.table.size}, value_size={p3_setup.table.value_size}, "
                "secret=False; setup is public protocol metadata plus encrypted projection payloads",
            )
        if p3_cache_hit:
            log("Server", OBLIVIOUS_FILTER_STAGE, f"setup_cache=hit key={real_inputs['p3_setup_cache_key']}")
        else:
            server.send(p3_setup)
            if SETUP_CACHE_ENABLED and real_inputs is not None:
                p3_cache_paths = _setup_cache_paths("p3", real_inputs["p3_setup_cache_key"])
                _save_pickle(p3_cache_paths["server"], p1_server)
                log("Server", OBLIVIOUS_FILTER_STAGE, f"setup_cache=miss saved={p3_cache_paths['server']}")
        p3_message = server.receive()
        if PROTOCOL_AUDIT:
            audit_log(
                "Server",
                OBLIVIOUS_FILTER_STAGE,
                f"client message type={type(p3_message).__name__}, encrypted_secret_count={len(p3_message.shuffled_secret_ciphertexts)}, "
                f"sample_ciphertexts={[hex(value)[:18] for value in p3_message.shuffled_secret_ciphertexts[:min(AUDIT_SAMPLE, len(p3_message.shuffled_secret_ciphertexts))]]}, "
                "secret=True; values are Paillier ciphertexts",
            )
        p1_candidates = p1_server.recover_candidates(p3_message, num_docs=NUM_DOCS)
        p3_candidate_indices = list(p1_candidates.candidate_indices)
        if not p1_candidates.candidates:
            log("Server", OBLIVIOUS_FILTER_STAGE, "candidate set empty; semantic branch will be skipped")
        candidate_mask_plain = p1_candidates.candidate_mask.to(DEVICE)
        update_metrics(
            "oblivious_filter",
            {
                "Server.candidate_count": int(candidate_mask_plain.sum().item()),
                "Server.encrypted_secret_count": int(len(p3_message.shuffled_secret_ciphertexts)),
                "Server.projection_collision_count": int(getattr(p3_setup, "projection_collision_count", 0)),
                "Server.okvs_slots": int(p3_setup.table.size),
                "Server.projection_count": int(len(p3_setup.masks)),
            },
        )
        server.send(candidate_mask_plain.cpu())
        semantic_candidate_ids, semantic_candidate_valid_mask, semantic_candidate_embeddings_plain = semantic_candidates_from_ids(
            db_embeddings,
            p3_candidate_indices,
        )
        semantic_top_k = min(TOP_K, int(semantic_candidate_ids.numel()))
        s_semantic_candidates_local, s_semantic_candidates_remote = share_data(semantic_candidate_embeddings_plain)
        semantic_candidate_embedding_share = s_semantic_candidates_local[0]
        server.send(semantic_candidate_ids.detach().cpu())
        server.send(s_semantic_candidates_remote)
        audit_value(
            "Server",
            OBLIVIOUS_FILTER_STAGE,
            "D_prime.candidate_mask",
            candidate_mask_plain,
            plaintext_visibility="S learns candidate set by ∏Oblivious Filter output",
        )
        log(
            "Server",
            OBLIVIOUS_FILTER_STAGE,
            f"candidates={len(p3_candidate_indices)}/{NUM_DOCS}, okvs_slots={p3_setup.table.size}, "
            f"candidate_rows_for_semantic={int(semantic_candidate_ids.numel())}, "
            f"semantic_top_k={semantic_top_k}, "
            f"paillier_key_size={P3_PAILLIER_KEY_SIZE}",
        )
        debug("Server", OBLIVIOUS_FILTER_STAGE, f"candidate_indices={p3_candidate_indices}")

        checkpoint.mark(OBLIVIOUS_FILTER_STAGE)
        # Protocol 2 / ∏PrivateBM25 Line 1 invokes Protocol 4 / ∏MultLPSI.
        p4_cache_hit = False
        if real_inputs is not None and real_inputs.get("p4_setup_cache_ready"):
            p4_cache_paths = _setup_cache_paths("p4", real_inputs["p4_setup_cache_key"])
            p4_bundle = _load_pickle(p4_cache_paths["server"])
            p4_server = p4_bundle["server"]
            p4_setup = p4_bundle["setup"]
            p2_server = Protocol2Server(config=RAG_CONFIG, protocol4=p4_server)
            p2_server.setup = p4_setup
            p4_cache_hit = True
        else:
            p2_server = Protocol2Server.paper_defaults(
                config=RAG_CONFIG,
                okvs_expansion=P4_OKVS_EXPANSION,
                oprf_secret_key=P4_OPRF_SECRET_KEY,
            )
            if real_inputs is not None and "document_tf_entries" in real_inputs:
                p4_setup = p2_server.build_lpsi_setup_from_entries(
                    num_docs=NUM_DOCS,
                    entries=real_inputs["document_tf_entries"],
                )
            else:
                p4_setup = p2_server.build_lpsi_setup(document_tf_plain.cpu())
        if PROTOCOL_AUDIT:
            audit_log(
                "Server",
                MULTLPSI_STAGE,
                f"setup type={type(p4_setup).__name__}, okvs_slots={p4_setup.table.size}, "
                f"value_size={p4_setup.table.value_size}, num_docs={p4_setup.num_docs}, "
                "secret=False; OKVS table stores encrypted labels",
            )
        if p4_cache_hit:
            log("Server", MULTLPSI_STAGE, f"setup_cache=hit key={real_inputs['p4_setup_cache_key']}")
        else:
            server.send(p4_setup)
            if SETUP_CACHE_ENABLED and real_inputs is not None:
                p4_cache_paths = _setup_cache_paths("p4", real_inputs["p4_setup_cache_key"])
                _save_pickle(p4_cache_paths["server"], {"setup": p4_setup, "server": p2_server.protocol4})
                log("Server", MULTLPSI_STAGE, f"setup_cache=miss saved={p4_cache_paths['server']}")
        p4_request = server.receive()
        if PROTOCOL_AUDIT:
            audit_log(
                "Server",
                MULTLPSI_STAGE,
                f"OPRF request type={type(p4_request).__name__}, count={len(p4_request.elements)}, "
                f"sample_blinded={[hex(value)[:18] for value in p4_request.elements[:min(AUDIT_SAMPLE, len(p4_request.elements))]]}, "
                "secret=True; server sees blinded OPRF elements",
            )
        p4_response = p2_server.evaluate_lpsi(p4_request)
        server.send(p4_response)
        log(
            "Server",
            MULTLPSI_STAGE,
            f"okvs_slots={p4_setup.table.size}, query_terms={len(p4_request.elements)}, "
            f"value_size={p4_setup.table.value_size}, okvs_method={p4_setup.table.method}, "
            f"okvs_expansion={P4_OKVS_EXPANSION}",
        )

        checkpoint.mark(MULTLPSI_STAGE)
        weighted_tf_remote = server.receive()
        tf_remote = server.receive()
        weighted_tf_share = weighted_tf_remote[0]
        tf_share = tf_remote[0]
        query_tokens_for_bm25 = query_bm25_tokens_for_debug.to(document_tf_plain.device)
        expected_tf_plain = document_tf_plain[query_tokens_for_bm25].T.to(DEVICE)
        expected_bm25 = expected_bm25_from_tf(expected_tf_plain, document_lengths_plain.to(DEVICE))
        expected_weighted_tf = (
            expected_bm25.idf.to(DEVICE).unsqueeze(0)
            * (RAG_CONFIG.bm25_k1 + 1.0)
            * expected_tf_plain
        )
        length_norm_plain = p2_server.length_norm_plain(document_lengths_plain)
        s_length_norm_local, s_length_norm_remote = share_data(length_norm_plain)
        server.send(s_length_norm_remote)
        length_norm_share = s_length_norm_local[0]
        audit_ass_pair(
            "Server",
            server,
            "Protocol2-bm25",
            "weighted_tf_numerator",
            weighted_tf_share,
            expected=expected_weighted_tf,
        )
        audit_ass_pair("Server", server, "Protocol2-bm25", "tf_i_j", tf_share, expected=expected_tf_plain)
        audit_ass_pair(
            "Server",
            server,
            "Protocol2-bm25",
            "length_norm",
            length_norm_share,
            expected=length_norm_plain,
        )
        p2_bm25_result = p2_server.score_from_shares(
            weighted_tf_share,
            tf_share,
            length_norm_share,
        )
        lexical_scores_share = p2_bm25_result.scores
        lexical_contrib_share = p2_bm25_result.contributions
        audit_ass_pair(
            "Server",
            server,
            "Protocol2-bm25",
            "BM25_contributions",
            lexical_contrib_share,
            expected=expected_bm25.contributions.to(DEVICE),
        )
        audit_ass_pair(
            "Server",
            server,
            "Protocol2-bm25",
            "BM25_scores",
            lexical_scores_share,
            expected=expected_bm25.scores.to(DEVICE),
        )
        log(
            "Server",
            "Protocol2-bm25",
            f"tf_share={tuple(tf_share.shape)}, score_share={tuple(lexical_scores_share.shape)}",
        )

        checkpoint.mark("Protocol2-bm25")
        semantic_topk_ids_for_audit = None
        lexical_topk_ids_for_audit = None
        if semantic_top_k:
            semantic_result = p1_server.finish_from_candidate_mask(
                query_emb_share,
                semantic_candidate_embedding_share,
                candidate_mask=semantic_candidate_valid_mask.cpu(),
                top_k=semantic_top_k,
                penalty=RAG_TOPK_PENALTY,
            )
            semantic_scores = semantic_result.scores
            expected_sem_scores = expected_semantic_scores(
                demo_semantic_query_embedding(),
                semantic_candidate_embeddings_plain,
                semantic_candidate_valid_mask,
            )
            expected_semantic_positions = torch.topk(expected_sem_scores, semantic_top_k).indices
        else:
            semantic_result = None
            semantic_scores = None
            expected_sem_scores = None
            expected_semantic_positions = None
        lexical_scores = lexical_scores_share
        lexical_result = p2_server.finish_from_scores(lexical_scores_share, top_k=TOP_K)
        lexical_indicators = lexical_result.indicators
        expected_lexical_ids = torch.topk(expected_bm25.scores.to(DEVICE), TOP_K).indices
        restored_semantic_indicators = None
        if semantic_top_k:
            restored_semantic_indicators = audit_ass_pair(
                "Server",
                server,
                "TopK",
                "semantic_scores_after_Oblivious_Filter",
                semantic_scores,
                expected=expected_sem_scores,
            )
            restored_semantic_indicators = audit_ass_pair(
                "Server",
                server,
                "TopK",
                "semantic_topk_indicators",
                semantic_result.indicators,
                expected=indicator_from_ids(expected_semantic_positions, semantic_top_k, int(semantic_candidate_ids.numel())),
            )
        audit_ass_pair(
            "Server",
            server,
            "TopK",
            "lexical_scores_BM25",
            lexical_scores,
            expected=expected_bm25.scores.to(DEVICE),
        )
        restored_lexical_indicators = audit_ass_pair(
            "Server",
            server,
            "TopK",
            "lexical_topk_indicators",
            lexical_indicators,
            expected=indicator_from_ids(expected_lexical_ids, TOP_K, NUM_DOCS),
        )
        semantic_topk_positions_for_audit = ids_from_indicator(restored_semantic_indicators) if restored_semantic_indicators is not None else None
        semantic_topk_ids_for_audit = (
            torch.tensor(
                map_candidate_positions_to_doc_ids(semantic_topk_positions_for_audit, semantic_candidate_ids),
                dtype=torch.long,
                device=DEVICE,
            )
            if semantic_topk_positions_for_audit is not None
            else None
        )
        lexical_topk_ids_for_audit = ids_from_indicator(restored_lexical_indicators) if restored_lexical_indicators is not None else None

        checkpoint.mark("TopK")
        pir_server_state = None
        if semantic_top_k:
            semantic_pir = p1_server.retrieve_topk_documents(
                server,
                db_embedding_payload,
                top_k=TOP_K,
                server_state=pir_server_state,
            )
            pir_server_state = semantic_pir.state
            log(
                "Server",
                "Suda-PIR",
                f"encrypted_query_bytes semantic={semantic_pir.request.get('query_bytes')}, "
                f"keys={'fresh' if 'keys' in semantic_pir.request else 'reused'}, scope=full-db",
            )
        else:
            semantic_pir = None
            log("Server", "Suda-PIR", "semantic skipped because D_prime is empty")
        lexical_pir = p2_server.retrieve_topk_documents(
            server,
            db_embedding_payload,
            top_k=TOP_K,
            server_state=pir_server_state,
        )
        pir_server_state = lexical_pir.state
        log(
            "Server",
            "Suda-PIR",
            f"encrypted_query_bytes lexical={lexical_pir.request.get('query_bytes')}, "
            f"keys={'fresh' if 'keys' in lexical_pir.request else 'reused'}",
        )

        my_doc_sem_share = semantic_pir.share if semantic_pir is not None else zero_document_share(0)
        my_doc_lex_share = lexical_pir.share
        expected_semantic_payload = None
        expected_lexical_payload = None
        if semantic_topk_positions_for_audit is not None and semantic_topk_ids_for_audit is not None:
            semantic_expected_ids = pad_doc_ids_for_fixed_pir(
                [int(value) for value in semantic_topk_ids_for_audit.detach().cpu().reshape(-1).tolist()],
                TOP_K,
                NUM_DOCS,
            )
            expected_semantic_payload = db_embedding_payload[semantic_expected_ids.to(dtype=torch.long).cpu()].to(DEVICE) / float(float_scale)
        if lexical_topk_ids_for_audit is not None:
            expected_lexical_payload = (
                db_embedding_payload[lexical_topk_ids_for_audit.to(dtype=torch.long).cpu()].to(DEVICE)
                / float(float_scale)
            )
        audit_ass_pair(
            "Server",
            server,
            "Suda-PIR",
            "semantic_retrieved_embedding",
            my_doc_sem_share,
            expected=expected_semantic_payload,
        )
        audit_ass_pair(
            "Server",
            server,
            "Suda-PIR",
            "lexical_retrieved_embedding",
            my_doc_lex_share,
            expected=expected_lexical_payload,
        )
        my_doc_sem_share = pad_ass_rows(my_doc_sem_share, TOP_K)
        semantic_pir_audit = semantic_pir.audit if semantic_pir is not None else None
        lexical_pir_audit = lexical_pir.audit
        print_topk_audit("Server", "Semantic", semantic_result.topk_audit if semantic_result is not None else None)
        print_topk_audit("Server", "Lexical", lexical_result.topk_audit)
        print_pir_audit("Server", "Semantic", semantic_pir_audit)
        print_pir_audit("Server", "Lexical", lexical_pir_audit)
        update_metrics(
            "pir",
            {
                "Server.semantic": pir_audit_to_metrics(semantic_pir_audit),
                "Server.lexical": pir_audit_to_metrics(lexical_pir_audit),
            },
        )

        checkpoint.mark("Suda-PIR")
        server.dummy_model(model_for_dummy)

        checkpoint.mark("Setup-joint-model")
        # Query word embeddings are computed with the same joint-sequence matmul shape as the dummy run.
        joint_batch = int(my_doc_sem_share.shape[0])
        my_query_share = query_word_embeddings_for_joint(model, sh_in[0], joint_batch, TOTAL_SEQ)
        my_joint_word_embeddings = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)
        audit_ass_pair("Server", server, "Fusion", "joint_word_embeddings", my_joint_word_embeddings)




        # 5). 接收 Client 发来的辅助张量 (Pos, Typ, Mask)
        my_pos_share = server.receive()[0]
        my_typ_share = server.receive()[0]
        mask = server.receive()
        # ---------------------------------------------------------
        # 5.执行联合推理
        # ---------------------------------------------------------
        seq_out, pool = bert_from_word_embeddings(model, my_joint_word_embeddings, my_pos_share, my_typ_share, mask)
        checkpoint.mark("Secure-BERT")
        audit_ass_pair("Server", server, "Secure-BERT", "pooler_output", pool)
        
        # 6. 还原结果
        c_pool = server.receive()
        final_pool = ArithmeticSecretSharing.restore_from_shares(pool, c_pool)
        final_pool_plain = final_pool.convert_to_real_field()
        pooler_first5 = final_pool_plain[:, :5].detach().cpu().tolist()
        update_metrics("final", {"pooler_first5": pooler_first5})
        log("Server", "Final", f"pooler_first5={pooler_first5}")
        finalize_total("Server", run_start)
                
    server.close()

# Single-query end-to-end protocol path, client side.
def run_client():
    run_start = time.perf_counter()
    checkpoint = CheckpointReporter("Client", run_start)
    client.online()
    checkpoint.mark("Connect")
    with PartyRuntime(client):
        print_pisces_contract("Client")
        check_model_aux_params("Client")
        check_native_runtime_paths("Client")
        checkpoint.mark("Preflight")
        # ---------------------------------------------------------
        # 1. 接收模型 (Encoder)
        # ---------------------------------------------------------
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        
        checkpoint.mark("Build-model")
        real_inputs = load_real_rag_inputs() if REAL_DATASET_MODE else None
        if real_inputs is None:
            ids = torch.tensor([[101, 7592, 2088, 102] + [0]*(SEQ-4)]).to(DEVICE)
            check_protocol2_aux_params("Client", NUM_DOCS, int(QUERY_BM25_TOKENS.numel()))
        else:
            ids = real_inputs["query_ids"]
            check_protocol2_aux_params("Client", NUM_DOCS, int(real_inputs["query_bm25_tokens"].numel()))
        pos = torch.arange(SEQ).unsqueeze(0).to(DEVICE)
        typ = torch.zeros_like(ids).to(DEVICE)
        mask = torch.ones_like(ids, dtype=torch.float32).to(DEVICE)
        
        oh_ids = F.one_hot(ids, BERT_CONFIG['vocab_size']).float()
        oh_pos = F.one_hot(pos, BERT_CONFIG['max_position_embeddings']).float()
        oh_typ = F.one_hot(typ, BERT_CONFIG['type_vocab_size']).float()

        checkpoint.mark("Prepare-query")
        dummy_ids_8 = torch.zeros(1, SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_8 = torch.zeros(1, SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_8 = torch.zeros(1, SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_8 = torch.ones(1, SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_8, dummy_pos_8, dummy_typ_8, dummy_mask_8)

        s_local = client.receive()
        model = load_model(model, s_local)


        checkpoint.mark("Setup-model")
        # ---------------------------------------------------------
        # 2. 接收服务端知识库的 Secret Share
        # ---------------------------------------------------------
        c_db_remote = client.receive()
        my_db_share = c_db_remote[0]
        log("Client", "Setup-database", f"db_share={tuple(my_db_share.shape)}")
        audit_ass_pair(
            "Client",
            client,
            "Setup-database",
            "Dv.document_embeddings",
            my_db_share,
            plaintext_visibility="received only as secret share",
        )

        checkpoint.mark("Setup-database")
        # ---------------------------------------------------------
        # 3. 发送 Query 并提取特征
        # ---------------------------------------------------------
        s_ids = share_data(oh_ids); client.send(s_ids[1])
        s_pos = share_data(oh_pos); client.send(s_pos[1])
        s_typ = share_data(oh_typ); client.send(s_typ[1])
        client.send(RingTensor.convert_to_ring(mask))
        audit_ass_pair("Client", client, "Query-encode", "Q.token_one_hot", s_ids[0][0], expected=oh_ids)
        audit_ass_pair("Client", client, "Query-encode", "Q.position_one_hot", s_pos[0][0], expected=oh_pos)
        audit_ass_pair("Client", client, "Query-encode", "Q.type_one_hot", s_typ[0][0], expected=oh_typ)
        
        _, pool = model(s_ids[0][0], s_pos[0][0], s_typ[0][0], RingTensor.convert_to_ring(mask))
        bert_query_emb_share = pool
        semantic_query_plain = demo_semantic_query_embedding()
        s_sem_query_local, s_sem_query_remote = share_data(semantic_query_plain)
        query_emb_share = s_sem_query_local[0]
        client.send(s_sem_query_remote)
        audit_ass_pair(
            "Client",
            client,
            "Query-encode",
            "q.semantic_embedding",
            query_emb_share,
            expected=semantic_query_plain,
        )
        log("Client", "Query-encode", f"query_ids={ids.detach().cpu().tolist()}")
        checkpoint.mark("Query-encode")
        # dummy_query_plain = torch.randn(1, 128).to(DEVICE)
        # s_query_local, s_query_remote = share_data(dummy_query_plain)
        # query_emb_share = s_query_local[0]
        # client.send(s_query_remote)

        # ---------------------------------------------------------
        # 4. RAG 核心流程 (参与距离计算 -> 参与召回)
        # ---------------------------------------------------------
        # Protocol 1 / ∏PrivateSS Line 1 invokes Protocol 3 / ∏Oblivious Filter.
        if real_inputs is not None and real_inputs.get("p3_setup_cache_ready"):
            p3_cache_paths = _setup_cache_paths("p3", real_inputs["p3_setup_cache_key"])
            p3_setup = _load_pickle(p3_cache_paths["client"])
            log("Client", OBLIVIOUS_FILTER_STAGE, f"setup_cache=hit key={real_inputs['p3_setup_cache_key']}")
        else:
            p3_setup = client.receive()
            if SETUP_CACHE_ENABLED and real_inputs is not None:
                p3_cache_paths = _setup_cache_paths("p3", real_inputs["p3_setup_cache_key"])
                _save_pickle(p3_cache_paths["client"], p3_setup)
                log("Client", OBLIVIOUS_FILTER_STAGE, f"setup_cache=miss saved={p3_cache_paths['client']}")
        if PROTOCOL_AUDIT:
            audit_log(
                "Client",
                OBLIVIOUS_FILTER_STAGE,
                f"received setup type={type(p3_setup).__name__}, masks={len(p3_setup.masks)}, "
                f"okvs_slots={p3_setup.table.size}, "
                "secret=False; encrypted projection payloads are opaque to C",
            )
        p1_client = Protocol1Client.paper_defaults(config=RAG_CONFIG)
        p3_message = p1_client.make_filter_query(semantic_query_plain.cpu(), p3_setup)
        p3_state = p1_client.protocol3.state
        if p3_state is None:
            raise RuntimeError("Protocol 3 client state was not populated")
        update_metrics(
            "oblivious_filter",
            {
                "Client.decoded_ciphertext_count": int(len(p3_state.decoded_ciphertexts)),
                "Client.encrypted_secret_count": int(len(p3_message.shuffled_secret_ciphertexts)),
                "Client.projection_collision_count": int(getattr(p3_setup, "projection_collision_count", 0)),
                "Client.okvs_slots": int(p3_setup.table.size),
                "Client.projection_count": int(len(p3_setup.masks)),
            },
        )
        client.send(p3_message)
        candidate_mask_plain = client.receive().to(DEVICE)
        semantic_candidate_ids = client.receive().to(dtype=torch.long, device=DEVICE)
        semantic_candidate_remote = client.receive()
        semantic_candidate_embedding_share = semantic_candidate_remote[0]
        semantic_candidate_valid_mask = torch.ones(
            int(semantic_candidate_ids.numel()),
            dtype=torch.float32,
            device=DEVICE,
        )
        semantic_top_k = min(TOP_K, int(semantic_candidate_ids.numel()))
        if PROTOCOL_AUDIT:
            audit_log(
                "Client",
                OBLIVIOUS_FILTER_STAGE,
                f"sent encrypted_secret_count={len(p3_message.shuffled_secret_ciphertexts)}, "
                f"decoded_ciphertexts={len(p1_client.protocol3.state.decoded_ciphertexts)}, "
                "client does not decrypt candidate secrets",
            )
        audit_value(
            "Client",
            OBLIVIOUS_FILTER_STAGE,
            "D_prime.candidate_mask",
            candidate_mask_plain,
            plaintext_visibility="audit sees S output forwarded for local scoring",
        )
        candidate_count = int(candidate_mask_plain.sum().item())
        log(
            "Client",
            OBLIVIOUS_FILTER_STAGE,
            f"candidates={candidate_count}/{NUM_DOCS}, projections={len(p3_setup.masks)}, "
            f"candidate_rows_for_semantic={int(semantic_candidate_ids.numel())}, "
            f"semantic_top_k={semantic_top_k}, "
            f"decoded_ciphertexts={len(p3_state.decoded_ciphertexts)}, "
            f"encrypted_secret_count={len(p3_message.shuffled_secret_ciphertexts)}, okvs_slots={p3_setup.table.size}",
        )
        debug("Client", OBLIVIOUS_FILTER_STAGE, f"candidate_mask={candidate_mask_plain.tolist()}")
        
        checkpoint.mark(OBLIVIOUS_FILTER_STAGE)
        # Protocol 2 / ∏PrivateBM25 Line 1 invokes Protocol 4 / ∏MultLPSI.
        query_tokens = real_inputs["query_bm25_tokens"] if real_inputs is not None else QUERY_BM25_TOKENS.to(DEVICE)
        if real_inputs is not None and real_inputs.get("p4_setup_cache_ready"):
            p4_cache_paths = _setup_cache_paths("p4", real_inputs["p4_setup_cache_key"])
            p4_setup = _load_pickle(p4_cache_paths["client"])
            log("Client", MULTLPSI_STAGE, f"setup_cache=hit key={real_inputs['p4_setup_cache_key']}")
        else:
            p4_setup = client.receive()
            if SETUP_CACHE_ENABLED and real_inputs is not None:
                p4_cache_paths = _setup_cache_paths("p4", real_inputs["p4_setup_cache_key"])
                _save_pickle(p4_cache_paths["client"], p4_setup)
                log("Client", MULTLPSI_STAGE, f"setup_cache=miss saved={p4_cache_paths['client']}")
        if PROTOCOL_AUDIT:
            audit_log(
                "Client",
                MULTLPSI_STAGE,
                f"received setup type={type(p4_setup).__name__}, okvs_slots={p4_setup.table.size}, "
                f"value_size={p4_setup.table.value_size}, num_docs={p4_setup.num_docs}",
            )
        p2_client = Protocol2Client.paper_defaults(config=RAG_CONFIG)
        p4_request = p2_client.make_lpsi_query(query_tokens.cpu())
        client.send(p4_request)
        p4_response = client.receive()
        tf_recovered = p2_client.recover_term_frequencies(p4_response, p4_setup).to(DEVICE)
        expected_tf_recovered = None
        if real_inputs is not None:
            expected_tf_recovered = real_inputs["document_tf_plain"][query_tokens].T.to(DEVICE)
        audit_value(
            "Client",
            MULTLPSI_STAGE,
            "tf_i_j_from_MultLPSI",
            tf_recovered,
            expected=expected_tf_recovered,
            plaintext_visibility="C learns term frequencies as Protocol 2 input",
        )
        log(
            "Client",
            MULTLPSI_STAGE,
            f"query_terms={query_tokens.tolist()}, tf_shape={tuple(tf_recovered.shape)}, "
            f"okvs_slots={p4_setup.table.size}, okvs_method={p4_setup.table.method}",
        )
        debug("Client", MULTLPSI_STAGE, f"tf_recovered={tf_recovered}")

        checkpoint.mark(MULTLPSI_STAGE)
        p2_weighted_tf = p2_client.weighted_tf_plain(tf_recovered, num_docs=p4_setup.num_docs)
        df = p2_weighted_tf.df
        idf = p2_weighted_tf.idf
        weighted_tf_plain = p2_weighted_tf.weighted_tf
        debug("Client", "Protocol2-bm25", f"df={df.tolist()}, idf={idf.tolist()}")
        debug("Client", "Protocol2-bm25", f"weighted_tf={weighted_tf_plain}")
        s_weighted_tf_local, s_weighted_tf_remote = share_data(weighted_tf_plain)
        s_tf_local, s_tf_remote = share_data(tf_recovered)
        client.send(s_weighted_tf_remote)
        client.send(s_tf_remote)
        length_norm_remote = client.receive()
        length_norm_share = length_norm_remote[0]
        audit_ass_pair(
            "Client",
            client,
            "Protocol2-bm25",
            "weighted_tf_numerator",
            s_weighted_tf_local[0],
            expected=weighted_tf_plain,
        )
        audit_ass_pair("Client", client, "Protocol2-bm25", "tf_i_j", s_tf_local[0], expected=tf_recovered)
        restored_length_norm = audit_ass_pair("Client", client, "Protocol2-bm25", "length_norm", length_norm_share)
        p2_bm25_result = p2_client.score_from_shares(
            s_weighted_tf_local[0],
            s_tf_local[0],
            length_norm_share,
        )
        lexical_scores_share = p2_bm25_result.scores
        lexical_contrib_share = p2_bm25_result.contributions
        expected_client_bm25 = None
        if restored_length_norm is not None:
            expected_client_contrib = weighted_tf_plain / (tf_recovered + restored_length_norm.to(DEVICE).unsqueeze(-1))
            expected_client_bm25 = expected_client_contrib.sum(dim=-1)
        else:
            expected_client_contrib = None
        audit_ass_pair(
            "Client",
            client,
            "Protocol2-bm25",
            "BM25_contributions",
            lexical_contrib_share,
            expected=expected_client_contrib,
        )
        audit_ass_pair(
            "Client",
            client,
            "Protocol2-bm25",
            "BM25_scores",
            lexical_scores_share,
            expected=expected_client_bm25,
        )
        log("Client", "Protocol2-bm25", f"df={df.detach().cpu().tolist()}, score_share={tuple(lexical_scores_share.shape)}")

        checkpoint.mark("Protocol2-bm25")
        if semantic_top_k:
            semantic_result = p1_client.finish_from_candidate_mask(
                query_emb_share,
                semantic_candidate_embedding_share,
                candidate_mask=semantic_candidate_valid_mask.cpu(),
                top_k=semantic_top_k,
                penalty=RAG_TOPK_PENALTY,
            )
            semantic_scores = semantic_result.scores
        else:
            semantic_result = None
            semantic_scores = None
        lexical_scores = lexical_scores_share
        lexical_result = p2_client.finish_from_scores(lexical_scores_share, top_k=TOP_K)
        lexical_indicators = lexical_result.indicators

        expected_sem_scores_client = None
        expected_semantic_indicator_client = None
        if real_inputs is not None and semantic_top_k:
            _, _, expected_semantic_candidate_embeddings = semantic_candidates_from_ids(
                real_inputs["db_embeddings"],
                semantic_candidate_ids,
            )
            expected_sem_scores_client = expected_semantic_scores(
                semantic_query_plain,
                expected_semantic_candidate_embeddings,
                semantic_candidate_valid_mask,
            )
            expected_semantic_indicator_client = indicator_from_ids(
                torch.topk(expected_sem_scores_client, semantic_top_k).indices,
                semantic_top_k,
                int(semantic_candidate_ids.numel()),
            )
        expected_lexical_indicator_client = None
        if expected_client_bm25 is not None:
            expected_lexical_indicator_client = indicator_from_ids(
                torch.topk(expected_client_bm25, TOP_K).indices,
                TOP_K,
                NUM_DOCS,
            )
        restored_semantic_indicators = None
        if semantic_top_k:
            audit_ass_pair(
                "Client",
                client,
                "TopK",
                "semantic_scores_after_Oblivious_Filter",
                semantic_scores,
                expected=expected_sem_scores_client,
            )
            restored_semantic_indicators = audit_ass_pair(
                "Client",
                client,
                "TopK",
                "semantic_topk_indicators",
                semantic_result.indicators,
                expected=expected_semantic_indicator_client,
            )
        audit_ass_pair("Client", client, "TopK", "lexical_scores_BM25", lexical_scores, expected=expected_client_bm25)
        restored_lexical_indicators = audit_ass_pair(
            "Client",
            client,
            "TopK",
            "lexical_topk_indicators",
            lexical_indicators,
            expected=expected_lexical_indicator_client,
        )
        if PROTOCOL_AUDIT:
            semantic_topk_positions = (
                ids_from_indicator(restored_semantic_indicators)
                if restored_semantic_indicators is not None
                else torch.empty(0, dtype=torch.long)
            )
            lexical_topk_ids = ids_from_indicator(restored_lexical_indicators)
        else:
            semantic_topk_positions = (
                client_owned_topk_ids(semantic_result.indicators)
                if semantic_result is not None
                else torch.empty(0, dtype=torch.long)
            )
            lexical_topk_ids = client_owned_topk_ids(lexical_indicators)
        semantic_ids_list = map_candidate_positions_to_doc_ids(semantic_topk_positions, semantic_candidate_ids)
        log(
            "Client",
            "TopK",
            f"semantic_positions={semantic_topk_positions.tolist()}, semantic_ids={semantic_ids_list}, "
            f"lexical_ids={lexical_topk_ids.tolist()}",
        )
        gold_ids = set(int(value) for value in real_inputs.get("gold_ids", [])) if real_inputs is not None else set()
        lexical_ids_list = [int(value) for value in lexical_topk_ids.detach().cpu().reshape(-1).tolist()]
        semantic_valid_ids_list = [doc_id for doc_id in semantic_ids_list if doc_id >= 0]
        dual_union_ids = list(dict.fromkeys(semantic_valid_ids_list + lexical_ids_list))
        semantic_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in semantic_valid_ids_list))
        lexical_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in lexical_ids_list))
        dual_union_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in dual_union_ids))
        log(
            "Client",
            "Retrieval-hit",
            f"gold_ids={sorted(gold_ids)}, semantic_hit={semantic_hit}, lexical_hit={lexical_hit}, "
            f"dual_union_hit={dual_union_hit}, dual_union_ids={dual_union_ids}",
        )
        update_metrics(
            "retrieval",
            {
                "semantic_ids": semantic_ids_list,
                "semantic_candidate_count": int(semantic_candidate_ids.numel()),
                "lexical_ids": lexical_ids_list,
                "dual_union_ids": dual_union_ids,
                "semantic_hit": semantic_hit,
                "lexical_hit": lexical_hit,
                "dual_union_hit": dual_union_hit,
                "semantic_topk_audit": topk_audit_to_metrics(semantic_result.topk_audit if semantic_result is not None else None),
                "lexical_topk_audit": topk_audit_to_metrics(lexical_result.topk_audit),
            },
        )

        checkpoint.mark("TopK")
        pir_client_state = None
        if semantic_top_k:
            semantic_pir_row_ids = pad_doc_ids_for_fixed_pir(semantic_valid_ids_list, TOP_K, NUM_DOCS)
            semantic_pir = p1_client.retrieve_topk_documents(
                client,
                semantic_pir_row_ids,
                dtype=torch.float32,
                device=DEVICE,
                previous_state=pir_client_state,
            )
            pir_client_state = semantic_pir.state
        else:
            semantic_pir = None
        lexical_pir = p2_client.retrieve_topk_documents(
            client,
            lexical_topk_ids,
            dtype=torch.float32,
            device=DEVICE,
            previous_state=pir_client_state,
        )
        pir_client_state = lexical_pir.state

        my_doc_sem_share = semantic_pir.share if semantic_pir is not None else zero_document_share(0)
        my_doc_lex_share = lexical_pir.share
        semantic_pir_audit = semantic_pir.audit if semantic_pir is not None else None
        lexical_pir_audit = lexical_pir.audit
        audit_ass_pair("Client", client, "Suda-PIR", "semantic_retrieved_embedding", my_doc_sem_share)
        audit_ass_pair("Client", client, "Suda-PIR", "lexical_retrieved_embedding", my_doc_lex_share)
        my_doc_sem_share = pad_ass_rows(my_doc_sem_share, TOP_K)
        print_topk_audit("Client", "Semantic", semantic_result.topk_audit if semantic_result is not None else None)
        print_topk_audit("Client", "Lexical", lexical_result.topk_audit)
        print_pir_audit("Client", "Semantic", semantic_pir_audit)
        print_pir_audit("Client", "Lexical", lexical_pir_audit)
        update_metrics(
            "pir",
            {
                "Client.semantic": pir_audit_to_metrics(semantic_pir_audit),
                "Client.lexical": pir_audit_to_metrics(lexical_pir_audit),
            },
        )

        checkpoint.mark("Suda-PIR")
        # ---------------------------------------------------------
        # 5. 配合还原结果 (发送语义路的 Share 给 Server)
        # ---------------------------------------------------------
        #client.send(top_k_docs_sem_share)
        
        # ================== 【第二次假跑】 ==================
        dummy_ids_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_32 = torch.ones(TOP_K, TOTAL_SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_32, dummy_pos_32, dummy_typ_32, dummy_mask_32)
        checkpoint.mark("Setup-joint-model")
        # =========================================================        

        log(
            "Client",
            "Fusion",
            f"join query, semantic PIR share, and lexical PIR share; semantic_rows={semantic_top_k}/{TOP_K}",
        )

        # Query word embeddings are computed with the same joint-sequence matmul shape as the dummy run.
        joint_batch = int(my_doc_sem_share.shape[0])
        my_query_share = query_word_embeddings_for_joint(model, s_ids[0][0], joint_batch, TOTAL_SEQ)
        my_joint_word_embeddings = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)
        audit_ass_pair("Client", client, "Fusion", "joint_word_embeddings", my_joint_word_embeddings)




        # 5) 构造 56 长度的 Pos, Typ, Mask
        joint_pos = torch.arange(TOTAL_SEQ).unsqueeze(0).repeat(joint_batch, 1).to(DEVICE)
        
        # 核心：用 0 标识 Query，用 1 标识所有的 Document
        joint_typ = torch.cat([
            torch.zeros(1, QUERY_LEN),      # 前 8 个是 Query (Type 0)
            torch.ones(1, SEM_DOC_LEN),     # 中间 24 个是语义文档 (Type 1)
            torch.ones(1, LEX_DOC_LEN)      # 最后 24 个是 BM25文档 (Type 1)
        ], dim=1).repeat(joint_batch, 1).long().to(DEVICE)
        
        joint_mask = joint_attention_mask_for_fusion(joint_batch, semantic_top_k)

        oh_joint_pos = F.one_hot(joint_pos, BERT_CONFIG['max_position_embeddings']).float()
        oh_joint_typ = F.one_hot(joint_typ, BERT_CONFIG['type_vocab_size']).float()

        # 6) 分享辅助张量并发送
        s_pos_local, s_pos_remote = share_data(oh_joint_pos); client.send(s_pos_remote)
        s_typ_local, s_typ_remote = share_data(oh_joint_typ); client.send(s_typ_remote)
        client.send(RingTensor.convert_to_ring(joint_mask))

        my_pos_share = s_pos_local[0]
        my_typ_share = s_typ_local[0]

        # ---------------------------------------------------------
        # 5.执行联合推理
        # ---------------------------------------------------------
        seq_out, pool = bert_from_word_embeddings(
            model,
            my_joint_word_embeddings,
            my_pos_share,
            my_typ_share,
            RingTensor.convert_to_ring(joint_mask),
        )
        checkpoint.mark("Secure-BERT")
        audit_ass_pair("Client", client, "Secure-BERT", "pooler_output", pool)
        
        # 6.还原结果
        client.send(pool)
        finalize_total("Client", run_start)

    client.close()


# Retrieval-only benchmark path, server side. Setup is done once, then the
# dual-path retrieval protocols are repeated for many queries.
def run_server_batch_retrieval():
    if not REAL_DATASET_MODE:
        raise RuntimeError("PISCES_RAG_QUERY_COUNT>1 currently requires PISCES_RAG_REAL_DATASET=1")
    run_start = time.perf_counter()
    checkpoint = CheckpointReporter("Server", run_start)
    query_indices = get_real_query_indices(REAL_QUERY_COUNT)
    server.online()
    checkpoint.mark("Connect")
    with PartyRuntime(server):
        print_pisces_contract("Server")
        check_model_aux_params("Server")
        check_native_runtime_paths("Server")
        checkpoint.mark("Preflight")
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        model = load_bert_tiny_weights(model, "Server")
        model_for_dummy = SecBertModel(BERT_CONFIG)
        for param in model_for_dummy.parameters():
            param.requires_grad = False
        model_for_dummy = load_bert_tiny_weights(model_for_dummy, "Server")
        checkpoint.mark("Build-model")
        server.dummy_model(model_for_dummy)
        word_embedding_table_for_pir = model.embeddings.word_embeddings.weight.detach().cpu().clone()
        s_local, s_remote = share_model(model)
        server.send(s_remote)
        model = load_model(model, s_local)
        checkpoint.mark("Setup-model")
        base_inputs = load_real_rag_inputs(query_indices[0])
        db_embeddings = base_inputs["db_embeddings"]
        s_db_local, s_db_remote = share_data(db_embeddings)
        server.send(s_db_remote)
        my_db_share = s_db_local[0]
        document_tf_plain = base_inputs["document_tf_plain"]
        document_lengths_plain = document_tf_plain.sum(dim=0).clamp_min(1.0)
        check_protocol2_aux_params("Server", NUM_DOCS, int(REAL_QUERY_TERM_LIMIT or max(1, base_inputs["query_bm25_tokens"].numel())))
        db_tokens_ids = base_inputs["db_tokens_ids"]
        db_embedding_payload = torch.round(
            word_embedding_table_for_pir[db_tokens_ids.detach().cpu().long()] * int(float_scale)
        ).to(dtype=torch.float32)
        log(
            "Server",
            "Setup-database",
            f"batch_queries={len(query_indices)}, db_embeddings={tuple(db_embeddings.shape)}, "
            f"bm25_tf={tuple(document_tf_plain.shape)}, pir_payload={tuple(db_embedding_payload.shape)}",
        )
        p3_cache_hit = bool(base_inputs.get("p3_setup_cache_ready"))
        p4_cache_hit = bool(base_inputs.get("p4_setup_cache_ready"))
        if p3_cache_hit:
            p3_cache_paths = _setup_cache_paths("p3", base_inputs["p3_setup_cache_key"])
            p1_server = _load_pickle(p3_cache_paths["server"])
            p3_setup = p1_server.setup
            log("Server", "Setup-database", f"p3_setup_cache=hit key={base_inputs['p3_setup_cache_key']}")
        else:
            p1_server = Protocol1Server.paper_defaults(
                config=RAG_CONFIG,
                paillier_key_size=P3_PAILLIER_KEY_SIZE,
                threshold=P3_THRESHOLD,
                projection_count=P3_PROJECTION_COUNT,
                simhash_bits=P3_SIMHASH_BITS,
            )
            p3_setup = p1_server.build_filter_setup(db_embeddings.cpu(), chunks=list(range(NUM_DOCS)))
            server.send(p3_setup)
            if SETUP_CACHE_ENABLED:
                p3_cache_paths = _setup_cache_paths("p3", base_inputs["p3_setup_cache_key"])
                _save_pickle(p3_cache_paths["server"], p1_server)
                log("Server", "Setup-database", f"p3_setup_cache=miss saved={p3_cache_paths['server']}")

        if p4_cache_hit:
            p4_cache_paths = _setup_cache_paths("p4", base_inputs["p4_setup_cache_key"])
            p4_bundle = _load_pickle(p4_cache_paths["server"])
            p4_server = p4_bundle["server"]
            p4_setup = p4_bundle["setup"]
            p2_server = Protocol2Server(config=RAG_CONFIG, protocol4=p4_server)
            p2_server.setup = p4_setup
            log("Server", "Setup-database", f"p4_setup_cache=hit key={base_inputs['p4_setup_cache_key']}")
        else:
            p2_server = Protocol2Server.paper_defaults(
                config=RAG_CONFIG,
                okvs_expansion=P4_OKVS_EXPANSION,
                oprf_secret_key=P4_OPRF_SECRET_KEY,
            )
            p4_setup = p2_server.build_lpsi_setup_from_entries(
                num_docs=NUM_DOCS,
                entries=base_inputs["document_tf_entries"],
            )
            server.send(p4_setup)
            if SETUP_CACHE_ENABLED:
                p4_cache_paths = _setup_cache_paths("p4", base_inputs["p4_setup_cache_key"])
                _save_pickle(p4_cache_paths["server"], {"setup": p4_setup, "server": p2_server.protocol4})
                log("Server", "Setup-database", f"p4_setup_cache=miss saved={p4_cache_paths['server']}")

        checkpoint.mark("Setup-database")
        # Per-query protocol loop. timed_batch only records report fields for
        # stage averages; it does not change protocol messages or shares.
        lexical_pir_server_state = None
        batch_start = time.perf_counter()
        for query_no, query_index in enumerate(query_indices):
            real_inputs = load_real_rag_inputs(query_index)
            query_label = f"BatchQ{query_no}"
            with timed_batch("Server", query_no, query_index, "Query-encode"):
                sh_in = server.receive()
                sh_pos = server.receive()
                sh_type = server.receive()
                mask = server.receive()
                semantic_query_remote = server.receive()
                query_emb_share = semantic_query_remote[0]
            with timed_batch("Server", query_no, query_index, OBLIVIOUS_FILTER_STAGE):
                p3_message = server.receive()
                p1_candidates = p1_server.recover_candidates(p3_message, num_docs=NUM_DOCS)
                candidate_mask_plain = p1_candidates.candidate_mask.to(DEVICE)
                server.send(candidate_mask_plain.cpu())
                p3_candidate_indices = list(p1_candidates.candidate_indices)
                semantic_candidate_ids, semantic_candidate_valid_mask, semantic_candidate_embeddings_plain = semantic_candidates_from_ids(
                    db_embeddings,
                    p3_candidate_indices,
                )
                semantic_top_k = min(TOP_K, int(semantic_candidate_ids.numel()))
                s_semantic_candidates_local, s_semantic_candidates_remote = share_data(semantic_candidate_embeddings_plain)
                semantic_candidate_embedding_share = s_semantic_candidates_local[0]
                server.send(semantic_candidate_ids.detach().cpu())
                server.send(s_semantic_candidates_remote)
            with timed_batch("Server", query_no, query_index, MULTLPSI_STAGE):
                p4_request = server.receive()
                p4_response = p2_server.evaluate_lpsi(p4_request)
                server.send(p4_response)
            with timed_batch("Server", query_no, query_index, "Protocol2-bm25"):
                weighted_tf_remote = server.receive()
                tf_remote = server.receive()
                weighted_tf_share = weighted_tf_remote[0]
                tf_share = tf_remote[0]
                query_tokens_for_bm25 = real_inputs["query_bm25_tokens"].to(document_tf_plain.device)
                expected_tf_plain = document_tf_plain[query_tokens_for_bm25].T.to(DEVICE)
                expected_bm25 = expected_bm25_from_tf(expected_tf_plain, document_lengths_plain.to(DEVICE))
                length_norm_plain = p2_server.length_norm_plain(document_lengths_plain)
                s_length_norm_local, s_length_norm_remote = share_data(length_norm_plain)
                server.send(s_length_norm_remote)
                length_norm_share = s_length_norm_local[0]
                p2_bm25_result = p2_server.score_from_shares(
                    weighted_tf_share,
                    tf_share,
                    length_norm_share,
                )
                lexical_scores_share = p2_bm25_result.scores
            with timed_batch("Server", query_no, query_index, "TopK"):
                if semantic_top_k:
                    semantic_result = p1_server.finish_from_candidate_mask(
                        query_emb_share,
                        semantic_candidate_embedding_share,
                        candidate_mask=semantic_candidate_valid_mask.cpu(),
                        top_k=semantic_top_k,
                        penalty=RAG_TOPK_PENALTY,
                    )
                else:
                    semantic_result = None
                lexical_result = p2_server.finish_from_scores(lexical_scores_share, top_k=TOP_K)
            with timed_batch("Server", query_no, query_index, "Suda-PIR"):
                if semantic_top_k:
                    semantic_pir = p1_server.retrieve_topk_documents(
                        server,
                        db_embedding_payload,
                        top_k=TOP_K,
                        server_state=lexical_pir_server_state,
                    )
                    lexical_pir_server_state = semantic_pir.state
                lexical_pir = p2_server.retrieve_topk_documents(
                    server,
                    db_embedding_payload,
                    top_k=TOP_K,
                    server_state=lexical_pir_server_state,
                )
                lexical_pir_server_state = lexical_pir.state
            if query_no == 0 or (query_no + 1) % 10 == 0 or query_no + 1 == len(query_indices):
                log("Server", query_label, f"finished query_index={query_index} ({query_no + 1}/{len(query_indices)})")
        batch_elapsed = time.perf_counter() - batch_start
        update_metrics("timings", {"Server.Batch-retrieval": batch_elapsed})
        update_metrics("batch_summary", {"Server.batch_elapsed_seconds": batch_elapsed})
        log("Server", "Batch", f"retrieval-only batch done queries={len(query_indices)} elapsed={batch_elapsed:.2f}s")
        summarize_batch_metrics()
        finalize_total("Server", run_start)
    server.close()


# Retrieval-only benchmark path, client side. It mirrors the server batch loop
# and records hit information for each query.
def run_client_batch_retrieval():
    if not REAL_DATASET_MODE:
        raise RuntimeError("PISCES_RAG_QUERY_COUNT>1 currently requires PISCES_RAG_REAL_DATASET=1")
    run_start = time.perf_counter()
    checkpoint = CheckpointReporter("Client", run_start)
    query_indices = get_real_query_indices(REAL_QUERY_COUNT)
    client.online()
    checkpoint.mark("Connect")
    with PartyRuntime(client):
        print_pisces_contract("Client")
        check_model_aux_params("Client")
        check_native_runtime_paths("Client")
        checkpoint.mark("Preflight")
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        checkpoint.mark("Build-model")
        dummy_ids_8 = torch.zeros(1, SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_8 = torch.zeros(1, SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_8 = torch.zeros(1, SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_8 = torch.ones(1, SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_8, dummy_pos_8, dummy_typ_8, dummy_mask_8)
        s_local = client.receive()
        model = load_model(model, s_local)
        checkpoint.mark("Setup-model")
        base_inputs = load_real_rag_inputs(query_indices[0])
        c_db_remote = client.receive()
        my_db_share = c_db_remote[0]
        if base_inputs.get("p3_setup_cache_ready"):
            p3_cache_paths = _setup_cache_paths("p3", base_inputs["p3_setup_cache_key"])
            p3_setup = _load_pickle(p3_cache_paths["client"])
            log("Client", "Setup-database", f"p3_setup_cache=hit key={base_inputs['p3_setup_cache_key']}")
        else:
            p3_setup = client.receive()
            if SETUP_CACHE_ENABLED:
                p3_cache_paths = _setup_cache_paths("p3", base_inputs["p3_setup_cache_key"])
                _save_pickle(p3_cache_paths["client"], p3_setup)
                log("Client", "Setup-database", f"p3_setup_cache=miss saved={p3_cache_paths['client']}")

        if base_inputs.get("p4_setup_cache_ready"):
            p4_cache_paths = _setup_cache_paths("p4", base_inputs["p4_setup_cache_key"])
            p4_setup = _load_pickle(p4_cache_paths["client"])
            log("Client", "Setup-database", f"p4_setup_cache=hit key={base_inputs['p4_setup_cache_key']}")
        else:
            p4_setup = client.receive()
            if SETUP_CACHE_ENABLED:
                p4_cache_paths = _setup_cache_paths("p4", base_inputs["p4_setup_cache_key"])
                _save_pickle(p4_cache_paths["client"], p4_setup)
                log("Client", "Setup-database", f"p4_setup_cache=miss saved={p4_cache_paths['client']}")
        log("Client", "Setup-database", f"batch_queries={len(query_indices)}, db_share={tuple(my_db_share.shape)}")

        checkpoint.mark("Setup-database")
        p1_client = Protocol1Client.paper_defaults(config=RAG_CONFIG)
        p2_client = Protocol2Client.paper_defaults(config=RAG_CONFIG)
        # Per-query protocol loop. The update_batch_result call below is
        # experiment bookkeeping, separate from the secure retrieval messages.
        lexical_pir_client_state = None
        batch_start = time.perf_counter()
        for query_no, query_index in enumerate(query_indices):
            real_inputs = load_real_rag_inputs(query_index)
            ids = real_inputs["query_ids"]
            pos = torch.arange(SEQ).unsqueeze(0).to(DEVICE)
            typ = torch.zeros_like(ids).to(DEVICE)
            mask = torch.ones_like(ids, dtype=torch.float32).to(DEVICE)
            oh_ids = F.one_hot(ids, BERT_CONFIG['vocab_size']).float()
            oh_pos = F.one_hot(pos, BERT_CONFIG['max_position_embeddings']).float()
            oh_typ = F.one_hot(typ, BERT_CONFIG['type_vocab_size']).float()
            with timed_batch("Client", query_no, query_index, "Query-encode"):
                s_ids = share_data(oh_ids)
                s_pos = share_data(oh_pos)
                s_typ = share_data(oh_typ)
                client.send(s_ids[1])
                client.send(s_pos[1])
                client.send(s_typ[1])
                client.send(RingTensor.convert_to_ring(mask))
                semantic_query_plain = real_inputs["semantic_query_embedding"]
                s_sem_query_local, s_sem_query_remote = share_data(semantic_query_plain)
                query_emb_share = s_sem_query_local[0]
                client.send(s_sem_query_remote)
            with timed_batch("Client", query_no, query_index, OBLIVIOUS_FILTER_STAGE):
                p3_message = p1_client.make_filter_query(semantic_query_plain.cpu(), p3_setup)
                client.send(p3_message)
                candidate_mask_plain = client.receive().to(DEVICE)
                semantic_candidate_ids = client.receive().to(dtype=torch.long, device=DEVICE)
                semantic_candidate_remote = client.receive()
                semantic_candidate_embedding_share = semantic_candidate_remote[0]
                semantic_candidate_valid_mask = torch.ones(
                    int(semantic_candidate_ids.numel()),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                semantic_top_k = min(TOP_K, int(semantic_candidate_ids.numel()))
            with timed_batch("Client", query_no, query_index, MULTLPSI_STAGE):
                query_tokens = real_inputs["query_bm25_tokens"]
                p4_request = p2_client.make_lpsi_query(query_tokens.cpu())
                client.send(p4_request)
                p4_response = client.receive()
                tf_recovered = p2_client.recover_term_frequencies(p4_response, p4_setup).to(DEVICE)
            with timed_batch("Client", query_no, query_index, "Protocol2-bm25"):
                p2_weighted_tf = p2_client.weighted_tf_plain(tf_recovered, num_docs=p4_setup.num_docs)
                weighted_tf_plain = p2_weighted_tf.weighted_tf
                s_weighted_tf_local, s_weighted_tf_remote = share_data(weighted_tf_plain)
                s_tf_local, s_tf_remote = share_data(tf_recovered)
                client.send(s_weighted_tf_remote)
                client.send(s_tf_remote)
                length_norm_remote = client.receive()
                length_norm_share = length_norm_remote[0]
                p2_bm25_result = p2_client.score_from_shares(
                    s_weighted_tf_local[0],
                    s_tf_local[0],
                    length_norm_share,
                )
                lexical_scores_share = p2_bm25_result.scores
            with timed_batch("Client", query_no, query_index, "TopK"):
                if semantic_top_k:
                    semantic_result = p1_client.finish_from_candidate_mask(
                        query_emb_share,
                        semantic_candidate_embedding_share,
                        candidate_mask=semantic_candidate_valid_mask.cpu(),
                        top_k=semantic_top_k,
                        penalty=RAG_TOPK_PENALTY,
                    )
                    semantic_topk_positions = client_owned_topk_ids(semantic_result.indicators)
                else:
                    semantic_result = None
                    semantic_topk_positions = torch.empty(0, dtype=torch.long)
                lexical_result = p2_client.finish_from_scores(lexical_scores_share, top_k=TOP_K)
                lexical_topk_ids = client_owned_topk_ids(lexical_result.indicators)
                semantic_ids_list = map_candidate_positions_to_doc_ids(semantic_topk_positions, semantic_candidate_ids)
                semantic_valid_ids_list = [doc_id for doc_id in semantic_ids_list if doc_id >= 0]
                lexical_ids_list = [int(value) for value in lexical_topk_ids.detach().cpu().reshape(-1).tolist()]
                dual_union_ids = list(dict.fromkeys(semantic_valid_ids_list + lexical_ids_list))
                gold_ids = set(int(value) for value in real_inputs.get("gold_ids", []))
                semantic_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in semantic_valid_ids_list))
                lexical_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in lexical_ids_list))
                dual_union_hit = bool(gold_ids and any(doc_id in gold_ids for doc_id in dual_union_ids))
            with timed_batch("Client", query_no, query_index, "Suda-PIR"):
                if semantic_top_k:
                    semantic_pir_row_ids = pad_doc_ids_for_fixed_pir(semantic_valid_ids_list, TOP_K, NUM_DOCS)
                    semantic_pir = p1_client.retrieve_topk_documents(
                        client,
                        semantic_pir_row_ids,
                        dtype=torch.float32,
                        device=DEVICE,
                        previous_state=lexical_pir_client_state,
                    )
                    lexical_pir_client_state = semantic_pir.state
                lexical_pir = p2_client.retrieve_topk_documents(
                    client,
                    lexical_topk_ids,
                    dtype=torch.float32,
                    device=DEVICE,
                    previous_state=lexical_pir_client_state,
                )
                lexical_pir_client_state = lexical_pir.state
            semantic_topk_metrics = topk_audit_to_metrics(semantic_result.topk_audit if semantic_result is not None else None)
            lexical_topk_metrics = topk_audit_to_metrics(lexical_result.topk_audit)
            semantic_pir_metrics = pir_audit_to_metrics(semantic_pir.audit if semantic_top_k else None)
            lexical_pir_metrics = pir_audit_to_metrics(lexical_pir.audit)
            semantic_topk_gc_bytes = int(semantic_topk_metrics.get("gc_communication_bytes") or 0)
            lexical_topk_gc_bytes = int(lexical_topk_metrics.get("gc_communication_bytes") or 0)
            semantic_pir_query_bytes = int(semantic_pir_metrics.get("query_bytes") or 0)
            semantic_pir_response_bytes = int(semantic_pir_metrics.get("response_bytes") or 0)
            lexical_pir_query_bytes = int(lexical_pir_metrics.get("query_bytes") or 0)
            lexical_pir_response_bytes = int(lexical_pir_metrics.get("response_bytes") or 0)
            update_batch_result(
                query_no,
                {
                    "query_index": int(real_inputs["query_index"]),
                    "query": real_inputs["query_text"],
                    "gold_ids": sorted(gold_ids),
                    "semantic_ids": semantic_ids_list,
                    "semantic_candidate_count": int(semantic_candidate_ids.numel()),
                    "payload_pir_scope": "full-db",
                    "semantic_pir_sent_keys": None if semantic_top_k == 0 else ("keys" in semantic_pir.request),
                    "lexical_ids": lexical_ids_list,
                    "dual_union_ids": dual_union_ids,
                    "semantic_hit": semantic_hit,
                    "lexical_hit": lexical_hit,
                    "dual_union_hit": dual_union_hit,
                    "lexical_pir_sent_keys": "keys" in lexical_pir.request,
                    "semantic_topk_gc_bytes": semantic_topk_gc_bytes,
                    "lexical_topk_gc_bytes": lexical_topk_gc_bytes,
                    "topk_gc_total_bytes": semantic_topk_gc_bytes + lexical_topk_gc_bytes,
                    "semantic_pir_query_bytes": semantic_pir_query_bytes,
                    "semantic_pir_response_bytes": semantic_pir_response_bytes,
                    "lexical_pir_query_bytes": lexical_pir_query_bytes,
                    "lexical_pir_response_bytes": lexical_pir_response_bytes,
                    "suda_pir_query_bytes": semantic_pir_query_bytes + lexical_pir_query_bytes,
                    "suda_pir_response_bytes": semantic_pir_response_bytes + lexical_pir_response_bytes,
                },
            )
            if query_no == 0 or (query_no + 1) % 10 == 0 or query_no + 1 == len(query_indices):
                log(
                    "Client",
                    f"BatchQ{query_no}",
                    f"query_index={real_inputs['query_index']} semantic_hit={semantic_hit} "
                    f"lexical_hit={lexical_hit} dual_hit={dual_union_hit} ({query_no + 1}/{len(query_indices)})",
                )
        batch_elapsed = time.perf_counter() - batch_start
        update_metrics("timings", {"Client.Batch-retrieval": batch_elapsed})
        update_metrics("batch_summary", {"Client.batch_elapsed_seconds": batch_elapsed})
        summarize_batch_metrics()
        log("Client", "Batch", f"retrieval-only batch done queries={len(query_indices)} elapsed={batch_elapsed:.2f}s")
        finalize_total("Client", run_start)
    client.close()


if __name__ == "__main__":
    if os.environ.get("SKIP_GEN_PARAMS") == "1":
        log("Init", "Params", "SKIP_GEN_PARAMS=1, skip unconditional generation")
        ensure_aux_params_for_current_run()
    else:
        if AUTO_GEN_PARAMS:
            ensure_aux_params_for_current_run()
        else:
            gen_params()
    if PARAMS_ONLY:
        log("Init", "Params", "PISCES_RAG_PARAMS_ONLY=1, auxiliary parameter preparation finished; exit before protocol run")
        write_report_if_requested()
        raise SystemExit(0)

    thread_errors = []

    def run_checked(name, target):
        try:
            target()
        except BaseException as exc:
            thread_errors.append((name, exc))
            raise

    if REAL_QUERY_COUNT > 1:
        log("Init", "Batch", f"PISCES_RAG_QUERY_COUNT={REAL_QUERY_COUNT}, run retrieval-only batch mode")
        server_target = run_server_batch_retrieval
        client_target = run_client_batch_retrieval
    else:
        server_target = run_server
        client_target = run_client

    t1 = threading.Thread(target=run_checked, args=("server", server_target))
    t2 = threading.Thread(target=run_checked, args=("client", client_target))
    t1.start(); t2.start()
    t1.join(); t2.join()
    if thread_errors:
        name, exc = thread_errors[0]
        raise RuntimeError(f"{name} thread failed") from exc
    write_report_if_requested()
    log("Done", "Pisces-RAG", "finished")
