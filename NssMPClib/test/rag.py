import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
import sys
import pickle
import time
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
    default_average_length,
    secure_bm25_scores_from_shares,
)
from NssMPC.application.rag.pisces.pir import (
    suda_native_client_extract,
    suda_native_make_client_request,
    suda_native_make_layout,
    suda_native_server_answer,
)
from NssMPC.application.rag.pisces.protocol1 import Protocol1Client, Protocol1Server, protocol1_finish_from_candidate_mask
from NssMPC.application.rag.pisces.protocol3 import AdditivePaillier, Protocol3Client, Protocol3Server
from NssMPC.application.rag.pisces.protocol4 import Protocol4Client, Protocol4Server
from NssMPC.application.rag.pisces.secure_sorting import secure_top_k_indicators
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer

# ==========================================
# 1. 全局配置
# ==========================================
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
P3_SIMHASH_BITS = 32
P3_THRESHOLD = 2
P3_PROJECTION_COUNT = 8
P3_PAILLIER_KEY_SIZE = int(os.environ.get("PISCES_RAG_P3_PAILLIER_KEY_SIZE", "256"))
RAG_CONFIG = PiscesConfig(top_k=TOP_K, simhash_bits=P3_SIMHASH_BITS, hamming_threshold=P3_THRESHOLD)
REAL_DATASET_MODE = os.environ.get("PISCES_RAG_REAL_DATASET") == "1"
REAL_DATASET_NAME = os.environ.get("PISCES_RAG_DATASET", "squad_dev_v2")
REAL_QUERY_INDEX = int(os.environ.get("PISCES_RAG_QUERY_INDEX", "0"))
REAL_CACHE_DIR = Path(os.environ.get("PISCES_RAG_CACHE_DIR", "/home/adslab/pazika/pisces/.embedding-cache"))
REAL_SQUAD_PATH = Path(os.environ.get("PISCES_RAG_SQUAD_PATH", "data/squad/dev-v2.0.json"))
REAL_HOTPOT_PATH = Path(
    os.environ.get(
        "PISCES_RAG_HOTPOT_PATH",
        "/home/adslab/pazika/pisces/hotpot/hotpot_dev_distractor_v1.json",
    )
)
REAL_INPUTS = None
VERBOSE = os.environ.get("PISCES_RAG_VERBOSE") == "1"
SUPPRESS_NATIVE_LOGS = os.environ.get("PISCES_RAG_NATIVE_LOGS") != "1"
_NATIVE_LOG_LOCK = threading.Lock()
B2A_KEY_COUNT = int(os.environ.get("PISCES_RAG_B2A_KEY_COUNT", "1000000"))
OBLIVIOUS_FILTER_STAGE = "Oblivious-Filter"
MULTLPSI_STAGE = "MultLPSI"


def log(role, stage, message):
    print(f"[{role}][{stage}] {message}", flush=True)


def debug(role, stage, message):
    if VERBOSE:
        log(role, stage, message)


@contextmanager
def timed(role, stage):
    start = time.perf_counter()
    log(role, stage, "start")
    try:
        yield
    finally:
        log(role, stage, f"done in {time.perf_counter() - start:.2f}s")


@contextmanager
def suppress_native_output():
    """Hide noisy native C++ backend logs unless PISCES_RAG_NATIVE_LOGS=1."""
    if not SUPPRESS_NATIVE_LOGS:
        yield
        return
    with _NATIVE_LOG_LOCK:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def print_pisces_contract(role):
    log(
        role,
        "Config",
        f"docs={NUM_DOCS}, top_k={TOP_K}, doc_len={SEM_DOC_LEN}, "
        f"real_dataset={REAL_DATASET_MODE}, dataset={REAL_DATASET_NAME}, "
        f"verbose={VERBOSE}, native_logs={not SUPPRESS_NATIVE_LOGS}",
    )


def print_topk_audit(role, path, audit):
    msg = (
        f"algorithm={audit.algorithm}, backend={audit.backend}, network={audit.network}, "
        f"comparisons={audit.comparisons}, paper_backend_available={audit.paper_backend_available}"
    )
    if audit.gc_communication_bytes is not None:
        msg += f", gc_bytes={audit.gc_communication_bytes}, value_bits={audit.gc_value_bits}, port={audit.gc_port}"
    log(role, f"{path}-TopK", msg)
    if not audit.paper_backend_available:
        log(role, f"{path}-TopK", f"fallback={audit.backend_gap}")


def print_pir_audit(role, path, pir_or_audit):
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


def finite_field_encoded_share_to_float_ass(field_share, modulus, label):
    """Convert fixed-point signed integers shared over Suda Z_p to float ASS."""
    modulus = int(modulus)
    local = torch.as_tensor(field_share, dtype=torch.long, device=DEVICE)
    summed_share = ArithmeticSecretSharing(RingTensor(local, dtype="int", device=DEVICE))
    carry_share = summed_share >= modulus
    value_share = summed_share - carry_share * modulus
    sign_share = value_share >= ((modulus + 1) // 2)
    signed_share = value_share - sign_share * modulus
    log(label, "PIR-to-MPC", f"Z_p share -> ASS share, shape={tuple(local.shape)}, modulus={modulus}, scale={int(float_scale)}")
    return ArithmeticSecretSharing(RingTensor(signed_share.item.tensor.to(dtype=torch.long), dtype="float", device=DEVICE))


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


def audit_message(audit):
    return {
        "implementation": audit.implementation,
        "paper_backend": audit.paper_backend,
        "paper_backend_available": audit.paper_backend_available,
        "backend_gap": audit.backend_gap,
        "output_shape": audit.output_shape,
        "polynomial_modulus": audit.polynomial_modulus,
        "native_batch_size": audit.native_batch_size,
        "native_padded_database_size": audit.native_padded_database_size,
        "native_query_bytes": audit.native_query_bytes,
        "native_response_bytes": audit.native_response_bytes,
    }


def _aux_param_count(param_cls, party_id=0, saved_name=None):
    base_name = saved_name or param_cls.__name__
    path = Path(param_path) / param_cls.__name__ / f"{base_name}_{party_id}.pth"
    if not path.exists():
        return 0
    return len(param_cls.load(str(path)))


def protocol2_aux_requirements(num_docs, query_terms):
    denominator_elems = int(num_docs) * int(query_terms)
    return {
        "DivKey": denominator_elems,
        "B2AKey": 2 * int(SCALE_BIT) * denominator_elems,
    }


def check_protocol2_aux_params(role, num_docs, query_terms):
    requirements = protocol2_aux_requirements(num_docs, query_terms)
    available_b2a = _aux_param_count(B2AKey, party_id=0)
    available_div = _aux_param_count(DivKey, party_id=0)
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


def load_real_rag_inputs():
    global REAL_INPUTS
    if REAL_INPUTS is not None:
        return REAL_INPUTS

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

    query_index = REAL_QUERY_INDEX
    query_text, gold_ids = prepared.queries[query_index]
    if not all(gold_id < NUM_DOCS for gold_id in gold_ids):
        for index, (candidate_query, candidate_gold) in enumerate(prepared.queries):
            if all(gold_id < NUM_DOCS for gold_id in candidate_gold):
                query_index = index
                query_text = candidate_query
                gold_ids = candidate_gold
                break
        else:
            raise ValueError(f"no {REAL_DATASET_NAME} query has all gold chunks within NUM_DOCS={NUM_DOCS}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    query_ids = tokenizer(
        query_text,
        add_special_tokens=True,
        truncation=True,
        max_length=QUERY_LEN,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].to(dtype=torch.long)

    doc_token_rows = []
    for chunk in prepared.chunks[:NUM_DOCS]:
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

    corpus_terms = term_frequencies[:NUM_DOCS]
    vocabulary = {}
    for counter in corpus_terms:
        for term in counter:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)
    query_terms = tokenizer.tokenize(query_text)
    query_bm25_tokens = torch.tensor(
        [vocabulary[term] for term in query_terms if term in vocabulary],
        dtype=torch.long,
        device=DEVICE,
    )
    if query_bm25_tokens.numel() == 0:
        query_bm25_tokens = torch.zeros(0, dtype=torch.long, device=DEVICE)
    document_tf_plain = torch.zeros(max(1, len(vocabulary)), NUM_DOCS, dtype=torch.float32, device=DEVICE)
    for doc_id, counter in enumerate(corpus_terms):
        for term, frequency in counter.items():
            token_id = vocabulary.get(term)
            if token_id is not None:
                document_tf_plain[token_id, doc_id] = float(frequency)

    REAL_INPUTS = {
        "query_index": query_index,
        "query_text": query_text,
        "gold_ids": sorted(gold_ids),
        "db_embeddings": torch.tensor(np.asarray(chunk_embeddings[:NUM_DOCS]), dtype=torch.float32, device=DEVICE),
        "semantic_query_embedding": torch.tensor(
            np.asarray(query_embeddings[query_index]),
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0),
        "query_ids": query_ids.to(DEVICE),
        "db_tokens_ids": db_tokens_ids,
        "document_tf_plain": document_tf_plain,
        "query_bm25_tokens": query_bm25_tokens,
        "query_terms": query_terms,
    }
    log(
        "Data",
        "Load",
        f"dataset={REAL_DATASET_NAME}, query_index={query_index}, gold_ids={sorted(gold_ids)}, "
        f"query={query_text!r}, docs={NUM_DOCS}, embedding_dim={REAL_INPUTS['db_embeddings'].shape[-1]}, "
        f"bm25_vocab={len(vocabulary)}, query_bm25_hits={query_bm25_tokens.numel()}",
    )
    return REAL_INPUTS


def demo_semantic_query_embedding():
    if REAL_DATASET_MODE:
        return load_real_rag_inputs()["semantic_query_embedding"]
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260624)
    return torch.randn(1, BERT_CONFIG["hidden_size"], generator=generator, device=DEVICE)


def gen_params():
    log("Init", "Params", "generate auxiliary parameters")
    if not os.path.exists('data'): os.makedirs('data')
    AssMulTriples.gen_and_save(50000000, saved_name='2PCBeaver')
    DivKey.gen_and_save(100000)
    GeLUKey.gen_and_save(100000)
    TanhKey.gen_and_save(100000)
    #MatmulTriples.gen_and_save(10000)
    Wrap.gen_and_save(10000000)
    ReciprocalSqrtKey.gen_and_save(10000)
    SigmaDICFKey.gen_and_save(100000)
    B2AKey.gen_and_save(B2A_KEY_COUNT)
    log("Init", "Params", f"B2AKey count={B2A_KEY_COUNT}")
    log("Init", "Params", "done")

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

def run_server():
    run_start = time.perf_counter()
    server.online()
    with PartyRuntime(server):
        print_pisces_contract("Server")
        # ---------------------------------------------------------
        # 1. 准备模型 (Encoder)
        # ---------------------------------------------------------
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        
        model_for_dummy = SecBertModel(BERT_CONFIG)

        with timed("Server", "Setup-model"):
            server.dummy_model(model_for_dummy)

        word_embedding_table_for_pir = model.embeddings.word_embeddings.weight.detach().cpu().clone()
        s_local, s_remote = share_model(model)



        server.send(s_remote)
        model = load_model(model, s_local)
        
        # ---------------------------------------------------------
        # 2. 准备服务端知识库 (Documents Database)
        # ---------------------------------------------------------
        with timed("Server", "Setup-database"):
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

        # ---------------------------------------------------------
        # 3. 接收 Client Query 并提取特征
        # ---------------------------------------------------------
        with timed("Server", "Query-encode"):
            sh_in = server.receive()
            sh_pos = server.receive()
            sh_type = server.receive()
            mask = server.receive()
            _, pool = model(sh_in[0], sh_pos[0], sh_type[0], mask)
            bert_query_emb_share = pool # shape: [1, 128]
            semantic_query_remote = server.receive()
            query_emb_share = semantic_query_remote[0]
            debug("Server", "P3", "received semantic query embedding share")

        #query_emb_share = server.receive()[0]
        
        # ---------------------------------------------------------
        # 4. RAG 核心流程：双路召回 (Dual-Path Retrieval)
        # ---------------------------------------------------------
        
        # Semantic coarse matching: Protocol 3, ∏Oblivious Filter.
        with timed("Server", OBLIVIOUS_FILTER_STAGE):
            p1_server = Protocol1Server(
                config=RAG_CONFIG,
                protocol3=Protocol3Server(
                okvs=BinaryOKVS(expansion=3.0, seed=b"rag-protocol3-okvs"),
                he=AdditivePaillier(key_size=P3_PAILLIER_KEY_SIZE),
                threshold=P3_THRESHOLD,
                projection_count=P3_PROJECTION_COUNT,
                seed=b"rag-protocol3",
                simhash_bits=P3_SIMHASH_BITS,
                ),
            )
            p3_setup = p1_server.build_filter_setup(db_embeddings.cpu(), chunks=list(range(NUM_DOCS)))
            server.send(p3_setup)
            p3_message = server.receive()
            p1_candidates = p1_server.recover_candidates(p3_message, num_docs=NUM_DOCS)
            p3_candidate_indices = list(p1_candidates.candidate_indices)
            if not p1_candidates.candidates:
                log("Server", OBLIVIOUS_FILTER_STAGE, "candidate set empty; fallback to all documents")
            candidate_mask_plain = p1_candidates.candidate_mask.to(DEVICE)
            server.send(candidate_mask_plain.cpu())
            log(
                "Server",
                OBLIVIOUS_FILTER_STAGE,
                f"candidates={len(p3_candidate_indices)}/{NUM_DOCS}, okvs_slots={p3_setup.table.size}, "
                f"paillier_key_size={P3_PAILLIER_KEY_SIZE}",
            )
            debug("Server", OBLIVIOUS_FILTER_STAGE, f"candidate_indices={p3_candidate_indices}")

        # Lexical term-frequency retrieval: Protocol 4, ∏MultLPSI.
        with timed("Server", MULTLPSI_STAGE):
            p4_params = DHOPRFParams()
            p4_server = Protocol4Server(oprf_server=DHOPRFServer(params=p4_params))
            p4_setup = p4_server.build_setup(document_tf_plain.cpu())
            server.send(p4_setup)
            p4_request = server.receive()
            p4_response = p4_server.evaluate_oprf(p4_request)
            server.send(p4_response)
            log(
                "Server",
                MULTLPSI_STAGE,
                f"okvs_slots={p4_setup.table.size}, query_terms={len(p4_request.elements)}, "
                f"value_size={p4_setup.table.value_size}",
            )

        with timed("Server", "Protocol2-bm25"):
            weighted_tf_remote = server.receive()
            tf_remote = server.receive()
            weighted_tf_share = weighted_tf_remote[0]
            tf_share = tf_remote[0]
            length_norm_plain = RAG_CONFIG.bm25_k1 * (
                1.0
                - RAG_CONFIG.bm25_b
                + RAG_CONFIG.bm25_b * document_lengths_plain / default_average_length(document_lengths_plain)
            )
            s_length_norm_local, s_length_norm_remote = share_data(length_norm_plain)
            server.send(s_length_norm_remote)
            length_norm_share = s_length_norm_local[0]
            lexical_scores_share, lexical_contrib_share = secure_bm25_scores_from_shares(
                weighted_tf_share,
                tf_share,
                length_norm_share,
            )
            log(
                "Server",
                "Protocol2-bm25",
                f"tf_share={tuple(tf_share.shape)}, score_share={tuple(lexical_scores_share.shape)}",
            )

        with timed("Server", "TopK"):
            semantic_result = protocol1_finish_from_candidate_mask(
                query_emb_share,
                my_db_share,
                candidate_mask=candidate_mask_plain.cpu(),
                top_k=TOP_K,
            )
            semantic_scores = semantic_result.scores
            lexical_scores = lexical_scores_share
            lexical_topk = secure_top_k_indicators(lexical_scores_share, TOP_K, return_audit=True)
            lexical_indicators = lexical_topk.indicators

        with timed("Server", "Suda-PIR"):
            semantic_layout = suda_native_make_layout(
                database_size=int(db_embedding_payload.shape[0]),
                record_shape=tuple(int(dim) for dim in db_embedding_payload.shape[1:]),
                selected_count=TOP_K,
                batch_size=1024,
            )
            lexical_layout = dict(semantic_layout)
            server.send(semantic_layout)
            server.send(lexical_layout)
            semantic_request = server.receive()
            lexical_request = server.receive()
            log("Server", "Suda-PIR", f"encrypted_query_bytes semantic={semantic_request.get('query_bytes')}, lexical={lexical_request.get('query_bytes')}")

            with suppress_native_output():
                semantic_answer = suda_native_server_answer(db_embedding_payload, semantic_request)
            with suppress_native_output():
                lexical_answer = suda_native_server_answer(db_embedding_payload, lexical_request)
            server.send(semantic_answer.response_message)
            server.send(audit_message(semantic_answer.audit))
            server.send(lexical_answer.response_message)
            server.send(audit_message(lexical_answer.audit))

            my_doc_sem_share = finite_field_encoded_share_to_float_ass(
                semantic_answer.server_share,
                semantic_answer.audit.polynomial_modulus,
                "Server][Semantic",
            )
            my_doc_lex_share = finite_field_encoded_share_to_float_ass(
                lexical_answer.server_share,
                lexical_answer.audit.polynomial_modulus,
                "Server][Lexical",
            )
            semantic_pir_audit = semantic_answer.audit
            lexical_pir_audit = lexical_answer.audit
            print_topk_audit("Server", "Semantic", semantic_result.topk_audit)
            print_topk_audit("Server", "Lexical", lexical_topk.audit)
            print_pir_audit("Server", "Semantic", semantic_pir_audit)
            print_pir_audit("Server", "Lexical", lexical_pir_audit)

        with timed("Server", "Setup-joint-model"):
            server.dummy_model(model_for_dummy)

        # Query word embeddings are computed with the same joint-sequence matmul shape as the dummy run.
        joint_batch = int(my_doc_sem_share.shape[0])
        my_query_share = query_word_embeddings_for_joint(model, sh_in[0], joint_batch, TOTAL_SEQ)
        my_joint_word_embeddings = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)




        # 5). 接收 Client 发来的辅助张量 (Pos, Typ, Mask)
        my_pos_share = server.receive()[0]
        my_typ_share = server.receive()[0]
        mask = server.receive()
        # ---------------------------------------------------------
        # 5.执行联合推理
        # ---------------------------------------------------------
        with timed("Server", "Secure-BERT"):
            seq_out, pool = bert_from_word_embeddings(model, my_joint_word_embeddings, my_pos_share, my_typ_share, mask)
        
        # 6. 还原结果
        c_pool = server.receive()
        final_pool = ArithmeticSecretSharing.restore_from_shares(pool, c_pool)
        log("Server", "Final", f"pooler_first5={final_pool.convert_to_real_field()[:, :5].detach().cpu().tolist()}")
        log("Server", "Total", f"done in {time.perf_counter() - run_start:.2f}s")
                
    server.close()

def run_client():
    run_start = time.perf_counter()
    client.online()
    with PartyRuntime(client):
        print_pisces_contract("Client")
        # ---------------------------------------------------------
        # 1. 接收模型 (Encoder)
        # ---------------------------------------------------------
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        
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

        with timed("Client", "Setup-model"):
            dummy_ids_8 = torch.zeros(1, SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
            dummy_pos_8 = torch.zeros(1, SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
            dummy_typ_8 = torch.zeros(1, SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
            dummy_mask_8 = torch.ones(1, SEQ).to(DEVICE)
            client.dummy_model(dummy_ids_8, dummy_pos_8, dummy_typ_8, dummy_mask_8)

            s_local = client.receive()
            model = load_model(model, s_local)


        # ---------------------------------------------------------
        # 2. 接收服务端知识库的 Secret Share
        # ---------------------------------------------------------
        with timed("Client", "Setup-database"):
            c_db_remote = client.receive()
            my_db_share = c_db_remote[0]
            log("Client", "Setup-database", f"db_share={tuple(my_db_share.shape)}")

        # ---------------------------------------------------------
        # 3. 发送 Query 并提取特征
        # ---------------------------------------------------------
        with timed("Client", "Query-encode"):
            s_ids = share_data(oh_ids); client.send(s_ids[1])
            s_pos = share_data(oh_pos); client.send(s_pos[1])
            s_typ = share_data(oh_typ); client.send(s_typ[1])
            client.send(RingTensor.convert_to_ring(mask))
            
            _, pool = model(s_ids[0][0], s_pos[0][0], s_typ[0][0], RingTensor.convert_to_ring(mask))
            bert_query_emb_share = pool
            semantic_query_plain = demo_semantic_query_embedding()
            s_sem_query_local, s_sem_query_remote = share_data(semantic_query_plain)
            query_emb_share = s_sem_query_local[0]
            client.send(s_sem_query_remote)
            log("Client", "Query-encode", f"query_ids={ids.detach().cpu().tolist()}")
        # dummy_query_plain = torch.randn(1, 128).to(DEVICE)
        # s_query_local, s_query_remote = share_data(dummy_query_plain)
        # query_emb_share = s_query_local[0]
        # client.send(s_query_remote)

        # ---------------------------------------------------------
        # 4. RAG 核心流程 (参与距离计算 -> 参与召回)
        # ---------------------------------------------------------
        # Semantic coarse matching: Protocol 3, ∏Oblivious Filter.
        with timed("Client", OBLIVIOUS_FILTER_STAGE):
            p3_setup = client.receive()
            p1_client = Protocol1Client(protocol3=Protocol3Client(shuffle_seed=b"rag-protocol3-client"))
            p3_message = p1_client.make_filter_query(semantic_query_plain.cpu(), p3_setup)
            client.send(p3_message)
            candidate_mask_plain = client.receive().to(DEVICE)
            candidate_count = int(candidate_mask_plain.sum().item())
            log(
                "Client",
                OBLIVIOUS_FILTER_STAGE,
                f"candidates={candidate_count}/{NUM_DOCS}, decoded_buckets={len(p1_client.protocol3.state.decoded_buckets)}, "
                f"decoded_ciphertexts={len(p1_client.protocol3.state.decoded_ciphertexts)}, okvs_slots={p3_setup.table.size}",
            )
            debug("Client", OBLIVIOUS_FILTER_STAGE, f"candidate_mask={candidate_mask_plain.tolist()}")
        
        # Lexical term-frequency retrieval: Protocol 4, ∏MultLPSI.
        with timed("Client", MULTLPSI_STAGE):
            query_tokens = real_inputs["query_bm25_tokens"] if real_inputs is not None else QUERY_BM25_TOKENS.to(DEVICE)
            p4_setup = client.receive()
            p4_client = Protocol4Client(oprf_client=DHOPRFClient(params=DHOPRFParams()))
            p4_request = p4_client.make_query(query_tokens.cpu())
            client.send(p4_request)
            p4_response = client.receive()
            tf_recovered = p4_client.recover_term_frequencies(p4_response, p4_setup).to(DEVICE)
            log(
                "Client",
                MULTLPSI_STAGE,
                f"query_terms={query_tokens.tolist()}, tf_shape={tuple(tf_recovered.shape)}, okvs_slots={p4_setup.table.size}",
            )
            debug("Client", MULTLPSI_STAGE, f"tf_recovered={tf_recovered}")

        with timed("Client", "Protocol2-bm25"):
            df = (tf_recovered > 0).sum(dim=0).float()
            idf = torch.log1p((p4_setup.num_docs - df + 0.5) / (df + 0.5))
            weighted_tf_plain = idf.unsqueeze(0) * (RAG_CONFIG.bm25_k1 + 1.0) * tf_recovered
            debug("Client", "Protocol2-bm25", f"df={df.tolist()}, idf={idf.tolist()}")
            debug("Client", "Protocol2-bm25", f"weighted_tf={weighted_tf_plain}")
            s_weighted_tf_local, s_weighted_tf_remote = share_data(weighted_tf_plain)
            s_tf_local, s_tf_remote = share_data(tf_recovered)
            client.send(s_weighted_tf_remote)
            client.send(s_tf_remote)
            length_norm_remote = client.receive()
            length_norm_share = length_norm_remote[0]
            lexical_scores_share, lexical_contrib_share = secure_bm25_scores_from_shares(
                s_weighted_tf_local[0],
                s_tf_local[0],
                length_norm_share,
            )
            log("Client", "Protocol2-bm25", f"df={df.detach().cpu().tolist()}, score_share={tuple(lexical_scores_share.shape)}")

        with timed("Client", "TopK"):
            semantic_result = protocol1_finish_from_candidate_mask(
                query_emb_share,
                my_db_share,
                candidate_mask=candidate_mask_plain.cpu(),
                top_k=TOP_K,
            )
            semantic_scores = semantic_result.scores
            lexical_scores = lexical_scores_share
            lexical_topk = secure_top_k_indicators(lexical_scores_share, TOP_K, return_audit=True)
            lexical_indicators = lexical_topk.indicators

            semantic_topk_ids = client_owned_topk_ids(semantic_result.indicators)
            lexical_topk_ids = client_owned_topk_ids(lexical_indicators)
            log("Client", "TopK", f"semantic_ids={semantic_topk_ids.tolist()}, lexical_ids={lexical_topk_ids.tolist()}")

        with timed("Client", "Suda-PIR"):
            semantic_layout = client.receive()
            lexical_layout = client.receive()
            with suppress_native_output():
                semantic_client_state, semantic_request = suda_native_make_client_request(
                    semantic_topk_ids,
                    semantic_layout,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            with suppress_native_output():
                lexical_client_state, lexical_request = suda_native_make_client_request(
                    lexical_topk_ids,
                    lexical_layout,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            client.send(semantic_request)
            client.send(lexical_request)
            semantic_response = client.receive()
            semantic_pir_audit = client.receive()
            lexical_response = client.receive()
            lexical_pir_audit = client.receive()
            with suppress_native_output():
                semantic_client_answer = suda_native_client_extract(semantic_client_state, semantic_response)
            with suppress_native_output():
                lexical_client_answer = suda_native_client_extract(lexical_client_state, lexical_response)

            my_doc_sem_share = finite_field_encoded_share_to_float_ass(
                semantic_client_answer.client_share,
                semantic_client_state.modulus,
                "Client][Semantic",
            )
            my_doc_lex_share = finite_field_encoded_share_to_float_ass(
                lexical_client_answer.client_share,
                lexical_client_state.modulus,
                "Client][Lexical",
            )
            print_topk_audit("Client", "Semantic", semantic_result.topk_audit)
            print_topk_audit("Client", "Lexical", lexical_topk.audit)
            print_pir_audit("Client", "Semantic", semantic_pir_audit)
            print_pir_audit("Client", "Lexical", lexical_pir_audit)

        # ---------------------------------------------------------
        # 5. 配合还原结果 (发送语义路的 Share 给 Server)
        # ---------------------------------------------------------
        #client.send(top_k_docs_sem_share)
        
        # ================== 【第二次假跑】 ==================
        with timed("Client", "Setup-joint-model"):
            dummy_ids_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
            dummy_pos_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
            dummy_typ_32 = torch.zeros(TOP_K, TOTAL_SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
            dummy_mask_32 = torch.ones(TOP_K, TOTAL_SEQ).to(DEVICE)
            client.dummy_model(dummy_ids_32, dummy_pos_32, dummy_typ_32, dummy_mask_32)
        # =========================================================        

        log("Client", "Fusion", "join query, semantic PIR share, and lexical PIR share")

        # Query word embeddings are computed with the same joint-sequence matmul shape as the dummy run.
        joint_batch = int(my_doc_sem_share.shape[0])
        my_query_share = query_word_embeddings_for_joint(model, s_ids[0][0], joint_batch, TOTAL_SEQ)
        my_joint_word_embeddings = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)




        # 5) 构造 56 长度的 Pos, Typ, Mask
        joint_pos = torch.arange(TOTAL_SEQ).unsqueeze(0).repeat(joint_batch, 1).to(DEVICE)
        
        # 核心：用 0 标识 Query，用 1 标识所有的 Document
        joint_typ = torch.cat([
            torch.zeros(1, QUERY_LEN),      # 前 8 个是 Query (Type 0)
            torch.ones(1, SEM_DOC_LEN),     # 中间 24 个是语义文档 (Type 1)
            torch.ones(1, LEX_DOC_LEN)      # 最后 24 个是 BM25文档 (Type 1)
        ], dim=1).repeat(joint_batch, 1).long().to(DEVICE)
        
        joint_mask = torch.ones(joint_batch, TOTAL_SEQ, dtype=torch.float32).to(DEVICE)

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
        with timed("Client", "Secure-BERT"):
            seq_out, pool = bert_from_word_embeddings(
                model,
                my_joint_word_embeddings,
                my_pos_share,
                my_typ_share,
                RingTensor.convert_to_ring(joint_mask),
            )
        
        # 6.还原结果
        client.send(pool)
        log("Client", "Total", f"done in {time.perf_counter() - run_start:.2f}s")

    client.close()

if __name__ == "__main__":
    if os.environ.get("SKIP_GEN_PARAMS") == "1":
        log("Init", "Params", "SKIP_GEN_PARAMS=1, skip auxiliary parameter generation")
    else:
        gen_params()

    thread_errors = []

    def run_checked(name, target):
        try:
            target()
        except BaseException as exc:
            thread_errors.append((name, exc))
            raise

    t1 = threading.Thread(target=run_checked, args=("server", run_server))
    t2 = threading.Thread(target=run_checked, args=("client", run_client))
    t1.start(); t2.start()
    t1.join(); t2.join()
    if thread_errors:
        name, exc = thread_errors[0]
        raise RuntimeError(f"{name} thread failed") from exc
    log("Done", "Pisces-RAG", "finished")
