import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
import sys

# 引入你的环境
from NssMPC.config import DEVICE
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
from NssMPC.application.rag.pisces.pir import SudaPIRToSharePlaintextProtocolBackend, suda_pir_to_share
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
NUM_DOCS = 10  # 知识库的文档库大小
TOP_K = 1      # 我们想要召回的文档数量
QUERY_LEN = 8
SEM_DOC_LEN = 24  # 语义路召回的文档长度
LEX_DOC_LEN = 24  # BM25路召回的文档长度
# 最终送入模型的总长度 = Query + Doc1(语义) + Doc2(词汇) = 56
TOTAL_SEQ = QUERY_LEN + SEM_DOC_LEN + LEX_DOC_LEN
VOCAB_SIZE_BM25 = 100
DEBUG = False
QUERY_BM25_TOKENS = torch.tensor([5, 8])
P3_SIMHASH_BITS = 32
P3_THRESHOLD = 2
P3_PROJECTION_COUNT = 8
RAG_CONFIG = PiscesConfig(top_k=TOP_K, simhash_bits=P3_SIMHASH_BITS, hamming_threshold=P3_THRESHOLD)


def print_pisces_contract(role):
    print(f"[{role}][Pisces-status] rag.py is the two-party NssMPClib integration demo.")
    print(f"[{role}][Pisces-status] Protocol 3/4 use the current cryptographic implementations.")
    print(f"[{role}][Pisces-status] Protocol 2 BM25 scoring and top-k run on ASS shares.")
    print(
        f"[{role}][Pisces-status] Payload PIR follows the Pisces dataflow: client restores top-k indices, "
        "server keeps the plaintext payload database, and Suda PIR-to-share returns document shares."
    )
    print(
        f"[{role}][Pisces-status] rag.py keeps retrieved document shares as ASS shares for secure BERT; "
        "the optional share-to-HE handoff is intentionally skipped."
    )


def print_topk_audit(role, path, audit):
    print(
        f"[{role}][{path}-TopK-audit] algorithm={audit.algorithm}, backend={audit.backend}, "
        f"network={audit.network}, comparisons={audit.comparisons}, "
        f"paper_backend_available={audit.paper_backend_available}"
    )
    if not audit.paper_backend_available:
        print(f"[{role}][{path}-TopK-audit] fallback={audit.backend_gap}")


def print_pir_audit(role, path, pir_or_audit):
    audit = pir_or_audit.audit if hasattr(pir_or_audit, "audit") else pir_or_audit
    if isinstance(audit, dict):
        print(
            f"[{role}][{path}-PIR-audit] implementation={audit['implementation']}, "
            f"paper_backend={audit['paper_backend']}, paper_backend_available={audit['paper_backend_available']}, "
            f"output_shape={audit['output_shape']}"
        )
        print(f"[{role}][{path}-PIR-audit] backend_gap={audit['backend_gap']}")
        print(
            f"[{role}][{path}-PIR-to-MPC] Suda output shares are wrapped as ArithmeticSecretSharing shares "
            "for the following secure BERT inference."
        )
        return
    print(
        f"[{role}][{path}-PIR-audit] implementation={audit.implementation}, "
        f"paper_backend={audit.paper_backend}, paper_backend_available={audit.paper_backend_available}, "
        f"output_shape={audit.output_shape}"
    )
    print(f"[{role}][{path}-PIR-audit] backend_gap={audit.backend_gap}")
    print(
        f"[{role}][{path}-PIR-to-MPC] Suda output shares are wrapped as ArithmeticSecretSharing shares "
        "for the following secure BERT inference."
    )


def ass_from_plain_share(share_tensor):
    return ArithmeticSecretSharing(RingTensor.convert_to_ring(share_tensor.to(DEVICE)))


def audit_message(audit):
    return {
        "implementation": audit.implementation,
        "paper_backend": audit.paper_backend,
        "paper_backend_available": audit.paper_backend_available,
        "backend_gap": audit.backend_gap,
        "output_shape": audit.output_shape,
    }


def demo_semantic_query_embedding():
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260624)
    return torch.randn(1, BERT_CONFIG["hidden_size"], generator=generator, device=DEVICE)


def gen_params():
    print("=== [Init] 生成辅助参数 ===")
    if not os.path.exists('data'): os.makedirs('data')
    AssMulTriples.gen_and_save(50000000, saved_name='2PCBeaver')
    DivKey.gen_and_save(100000)
    GeLUKey.gen_and_save(100000)
    TanhKey.gen_and_save(100000)
    #MatmulTriples.gen_and_save(10000)
    Wrap.gen_and_save(10000000)
    ReciprocalSqrtKey.gen_and_save(10000)
    SigmaDICFKey.gen_and_save(100000)
    B2AKey.gen_and_save(100000)
    print("=== [Init] 参数生成完成 ===\n")

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

        print("[Server] 执行 Dummy Model 1 (Seq=8)...")
        server.dummy_model(model_for_dummy) 

        s_local, s_remote = share_model(model)



        server.send(s_remote)
        model = load_model(model, s_local)
        
        # ---------------------------------------------------------
        # 2. 准备服务端知识库 (Documents Database)
        # ---------------------------------------------------------
        print("[Server] 构建并分享密态知识库...")
        db_generator = torch.Generator(device=DEVICE)
        db_generator.manual_seed(20260625)
        db_embeddings = torch.randn(
            NUM_DOCS,
            BERT_CONFIG['hidden_size'],
            generator=db_generator,
            device=DEVICE,
        )
        db_embeddings[0] = demo_semantic_query_embedding()[0]
        
        s_db_local, s_db_remote = share_data(db_embeddings)
        server.send(s_db_remote)
        my_db_share = s_db_local[0] 

        print("[Server] 构建 Protocol 4 文档词频矩阵 (词汇路)...")
        document_tf_plain = torch.randint(
            0,
            4,
            (VOCAB_SIZE_BM25, NUM_DOCS),
            dtype=torch.float32,
            device=DEVICE,
        )
        document_lengths_plain = document_tf_plain.sum(dim=0).clamp_min(1.0)
        print(f"[Server][P2-debug] document lengths for BM25={document_lengths_plain.tolist()}")
        print(f"[Server][P2-debug] average document length={default_average_length(document_lengths_plain):.6f}")
        for token in QUERY_BM25_TOKENS.tolist():
            print(f"[Server][P4-debug] token={token} plaintext TF across docs: {document_tf_plain[token].tolist()}")


        #准备服务端的文档 Token 数据库 [NUM_DOCS, 24, 30522]
        print("[Server] 构建服务端明文文档 Token 数据库，用于 Pisces PIR-to-share...")
        # 为了演示，生成 10 篇随机的 Token ID 文档
        db_tokens_ids = torch.randint(
            0,
            BERT_CONFIG['vocab_size'],
            (NUM_DOCS, SEM_DOC_LEN),
            device=DEVICE,
        )
        db_tokens_onehot = F.one_hot(db_tokens_ids, BERT_CONFIG['vocab_size']).float()

        # ---------------------------------------------------------
        # 3. 接收 Client Query 并提取特征
        # ---------------------------------------------------------
        print("[Server] 等待 Client 输入 Query...")
        sh_in = server.receive()
        sh_pos = server.receive()
        sh_type = server.receive()
        mask = server.receive()

        print("[Server] 提取 Query 密态 Embedding...")
        _, pool = model(sh_in[0], sh_pos[0], sh_type[0], mask)
        bert_query_emb_share = pool # shape: [1, 128]
        semantic_query_remote = server.receive()
        query_emb_share = semantic_query_remote[0]
        print("[Server][P3-debug] received Client semantic query embedding share for Protocol 3/semantic fine scoring.")

        #query_emb_share = server.receive()[0]
        
        # ---------------------------------------------------------
        # 4. RAG 核心流程：双路召回 (Dual-Path Retrieval)
        # ---------------------------------------------------------
        
        print("[Server] RAG: 开始双路密态打分与召回...")

        # 【第一路：语义检索 coarse filter via Pisces Protocol 3】
        print("[Server] RAG: 执行语义路 Protocol 3 (SimHash projections + OKVS + Paillier/Shamir filter)...")
        p1_server = Protocol1Server(
            config=RAG_CONFIG,
            protocol3=Protocol3Server(
            okvs=BinaryOKVS(expansion=3.0, seed=b"rag-protocol3-okvs"),
            he=AdditivePaillier(key_size=64),
            threshold=P3_THRESHOLD,
            projection_count=P3_PROJECTION_COUNT,
            seed=b"rag-protocol3",
            simhash_bits=P3_SIMHASH_BITS,
            ),
        )
        p3_setup = p1_server.build_filter_setup(db_embeddings.cpu(), chunks=list(range(NUM_DOCS)))
        print(
            "[Server][P3-debug] setup: "
            f"simhash_bits={p3_setup.simhash_bits}, masks={len(p3_setup.masks)}, "
            f"projection_weight={p3_setup.projection_weight}, bucket_capacity={p3_setup.bucket_capacity}, "
            f"OKVS slots={p3_setup.table.size}, value_size={p3_setup.table.value_size}"
        )
        server.send(p3_setup)
        p3_message = server.receive()
        p1_candidates = p1_server.recover_candidates(p3_message, num_docs=NUM_DOCS)
        p3_candidate_indices = list(p1_candidates.candidate_indices)
        if not p1_candidates.candidates:
            print("[Server][P3-debug] candidate set is empty; demo falls back to all documents for semantic fine scoring.")
        candidate_mask_plain = p1_candidates.candidate_mask.to(DEVICE)
        server.send(candidate_mask_plain.cpu())
        print(f"[Server][P3-debug] candidate_indices={p3_candidate_indices}")

        
        # # ====== 插入测试代码 (Start) ======
        # # 接收客户端的分数Share，还原成明文，用 PyTorch 算一遍正确答案
        # c_scores_sem = server.receive()
        # plain_scores = ArithmeticSecretSharing.restore_from_shares(scores_sem_share, c_scores_sem).convert_to_real_field()
        # topk_scores_plain, topk_indices_plain = torch.topk(plain_scores, TOP_K) # 取 Top-1
        # gt_topk_doc = db_embeddings[topk_indices_plain] # 根据真实索引去原数据库拿文档
        
        # print("\n[DEBUG - 明文验证] 真实打分结果:", plain_scores)
        # print("[DEBUG - 明文验证] PyTorch 选出的 Top-1 分数:", topk_scores_plain)
        # print("[DEBUG - 明文验证] 对应的真实文档特征 (前5维):\n", gt_topk_doc[:, :5])



        # 【第二路：词汇检索 BM25 Path via Pisces Protocol 4】
        print("[Server] RAG: 执行词汇路 Protocol 4 (OPRF + OKVS + AES labels)...")
        p4_params = DHOPRFParams()
        p4_server = Protocol4Server(oprf_server=DHOPRFServer(params=p4_params))
        p4_setup = p4_server.build_setup(document_tf_plain.cpu())
        print(
            "[Server][P4-debug] OKVS setup: "
            f"num_docs={p4_setup.num_docs}, slots={p4_setup.table.size}, "
            f"value_size={p4_setup.table.value_size}, seed_prefix={p4_setup.table.seed[:8].hex()}"
        )
        server.send(p4_setup)
        p4_request = server.receive()
        print(f"[Server][P4-debug] received OPRF blind request count={len(p4_request.elements)}")
        if p4_request.elements:
            print(f"[Server][P4-debug] first blinded element prefix={hex(p4_request.elements[0])[:34]}")
        p4_response = p4_server.evaluate_oprf(p4_request)
        print(f"[Server][P4-debug] sending OPRF response count={len(p4_response.elements)}")
        server.send(p4_response)

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
        print(f"[Server][P2-debug] received weighted_tf share shape={weighted_tf_share.shape}")
        print(f"[Server][P2-debug] received tf share shape={tf_share.shape}")
        print(f"[Server][P2-debug] local length_norm share shape={length_norm_share.shape}")
        print(f"[Server][P2-debug] secure BM25 contribution share shape={lexical_contrib_share.shape}")
        print(f"[Server][P2-debug] secure BM25 score share shape={lexical_scores_share.shape}")

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

        server.send(semantic_result.indicators)
        server.send(lexical_indicators)
        semantic_indicators_plain = server.receive().cpu()
        lexical_indicators_plain = server.receive().cpu()
        print(f"[Server][Semantic-PIR-query] received client-restored top-k indicators shape={tuple(semantic_indicators_plain.shape)}")
        print(f"[Server][Lexical-PIR-query] received client-restored top-k indicators shape={tuple(lexical_indicators_plain.shape)}")

        pir_backend = SudaPIRToSharePlaintextProtocolBackend(modulus=65_537, seed=20260708)
        semantic_pir = suda_pir_to_share(semantic_indicators_plain, db_tokens_onehot.cpu(), backend=pir_backend)
        lexical_pir = suda_pir_to_share(lexical_indicators_plain, db_tokens_onehot.cpu(), backend=pir_backend)
        server.send(semantic_pir.client_share)
        server.send(audit_message(semantic_pir.audit))
        server.send(lexical_pir.client_share)
        server.send(audit_message(lexical_pir.audit))
        my_doc_sem_share = ass_from_plain_share(semantic_pir.server_share)
        my_doc_lex_share = ass_from_plain_share(lexical_pir.server_share)
        print(f"[Server][P3-debug] masked semantic_scores shape={semantic_scores.shape}")
        print(f"[Server][P4-debug] lexical_scores_share shape={lexical_scores.shape}")
        print_topk_audit("Server", "Semantic", semantic_result.topk_audit)
        print_topk_audit("Server", "Lexical", lexical_topk.audit)
        print_pir_audit("Server", "Semantic", semantic_pir)
        print_pir_audit("Server", "Lexical", lexical_pir)

        # ---------------------------------------------------------
        # 5. 还原结果进行验证 (这里验证一下语义路的结果)
        # ---------------------------------------------------------
        # c_top_k_docs = server.receive()
        # final_docs = ArithmeticSecretSharing.restore_from_shares(top_k_docs_sem_share, c_top_k_docs)
        # print("\n=== [Server] RAG 语义路召回的文档明文 (前两维) ===")
        # print(final_docs.convert_to_real_field()[:, :5])
        
        print("[Server] 执行 Dummy Model 2 (Seq=56)...")
        server.dummy_model(model_for_dummy)
        
        print("[融合] 拼接 Semantic 和 BM25 召回的密态文档...")
        
        # # 1). 模拟将【语义路】召回的文档转为密文 Share
        # doc_sem_ids = torch.tensor([[666, 777, 888, 999, 102] + [0]*(SEM_DOC_LEN-5)]).to(DEVICE)
        # oh_doc_sem = F.one_hot(doc_sem_ids, BERT_CONFIG['vocab_size']).float()
        # s_doc_sem_local, s_doc_sem_remote = share_data(oh_doc_sem)
        # server.send(s_doc_sem_remote)
        # my_doc_sem_share = s_doc_sem_local[0]  # [1, 24, V]

        # # 2). 模拟将【词汇路(BM25)】召回的文档转为密文 Share
        # doc_lex_ids = torch.tensor([[111, 222, 333, 444, 102] + [0]*(LEX_DOC_LEN-5)]).to(DEVICE)
        # oh_doc_lex = F.one_hot(doc_lex_ids, BERT_CONFIG['vocab_size']).float()
        # s_doc_lex_local, s_doc_lex_remote = share_data(oh_doc_lex)
        # server.send(s_doc_lex_remote)
        # my_doc_lex_share = s_doc_lex_local[0]  # [1, 24, V]

        # # 3). 拿到 Client 之前发来的 Query Share (长度 8)
        # my_query_share = sh_in[0]

        # # 4). 密态无缝拼接 Query + SemDoc + LexDoc
        # print("[Server] 密态拼接 Query 和 双路 Document...")
        # # 把三者在序列维度(dim=1)拼接，总长 8 + 24 + 24 = 56
        # my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)



        print("[融合] 使用 Suda PIR-to-share 输出的密态文档拼接真实 Token 序列...")

        # 3). 拿到 Client 发来的 Query Share，直接拼接！
        my_query_share = sh_in[0]
        my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)




        # 5). 接收 Client 发来的辅助张量 (Pos, Typ, Mask)
        my_pos_share = server.receive()[0]
        my_typ_share = server.receive()[0]
        mask = server.receive()
        # ---------------------------------------------------------
        # 5.执行联合推理
        # ---------------------------------------------------------
        print("[Server] 执行联合推理...")
        seq_out, pool = model(my_joint_ids_share, my_pos_share, my_typ_share, mask)
        
        # 6. 还原结果
        c_pool = server.receive()
        final_pool = ArithmeticSecretSharing.restore_from_shares(pool, c_pool)
        print("\n=== [Server] 联合推理 Pooler 输出 ===")
        print(final_pool.convert_to_real_field()[:, :5])
                
    server.close()

def run_client():
    client.online()
    with PartyRuntime(client):
        print_pisces_contract("Client")
        # ---------------------------------------------------------
        # 1. 接收模型 (Encoder)
        # ---------------------------------------------------------
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        
        ids = torch.tensor([[101, 7592, 2088, 102] + [0]*(SEQ-4)]).to(DEVICE)
        pos = torch.arange(SEQ).unsqueeze(0).to(DEVICE)
        typ = torch.zeros_like(ids).to(DEVICE)
        mask = torch.ones_like(ids, dtype=torch.float32).to(DEVICE)
        
        oh_ids = F.one_hot(ids, BERT_CONFIG['vocab_size']).float()
        oh_pos = F.one_hot(pos, BERT_CONFIG['max_position_embeddings']).float()
        oh_typ = F.one_hot(typ, BERT_CONFIG['type_vocab_size']).float()

        # print("[Client] 执行 Dummy Model...")
        # client.dummy_model(oh_ids, oh_pos, oh_typ, mask)
        print("[Client] 执行 Dummy Model 1 (Seq=8)...")
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
        print("[Client] 接收密态知识库...")
        c_db_remote = client.receive()
        my_db_share = c_db_remote[0]

        # ---------------------------------------------------------
        # 3. 发送 Query 并提取特征
        # ---------------------------------------------------------
        print("[Client] 发送并编码 Query...")
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
        print("[Client][P3-debug] shared semantic query embedding for Protocol 3/semantic fine scoring.")
        # dummy_query_plain = torch.randn(1, 128).to(DEVICE)
        # s_query_local, s_query_remote = share_data(dummy_query_plain)
        # query_emb_share = s_query_local[0]
        # client.send(s_query_remote)

        # ---------------------------------------------------------
        # 4. RAG 核心流程 (参与距离计算 -> 参与召回)
        # ---------------------------------------------------------
        print("[Client] RAG: 参与双路密态打分与召回...")

        # 【第一路：语义检索 coarse filter via Pisces Protocol 3】
        print("[Client] RAG: 执行语义路 Protocol 3 (SimHash projections + OKVS + Paillier/Shamir filter)...")
        p3_setup = client.receive()
        print(
            "[Client][P3-debug] received setup: "
            f"simhash_bits={p3_setup.simhash_bits}, masks={len(p3_setup.masks)}, "
            f"projection_weight={p3_setup.projection_weight}, bucket_capacity={p3_setup.bucket_capacity}, "
            f"OKVS slots={p3_setup.table.size}, value_size={p3_setup.table.value_size}"
        )
        p1_client = Protocol1Client(protocol3=Protocol3Client(shuffle_seed=b"rag-protocol3-client"))
        p3_message = p1_client.make_filter_query(semantic_query_plain.cpu(), p3_setup)
        client.send(p3_message)
        candidate_mask_plain = client.receive().to(DEVICE)
        print(
            f"[Client][P3-debug] decoded buckets={len(p1_client.protocol3.state.decoded_buckets)}, "
            f"decoded ciphertexts={len(p1_client.protocol3.state.decoded_ciphertexts)}, "
            f"candidate_mask={candidate_mask_plain.tolist()}"
        )
        
        # 【第二路：词汇检索 BM25 Path via Pisces Protocol 4】
        print("[Client] RAG: 执行词汇路 Protocol 4 (OPRF + OKVS + AES labels)...")
        query_tokens = QUERY_BM25_TOKENS.to(DEVICE)
        print(f"[Client][P4-debug] query tokens={query_tokens.tolist()}")
        p4_setup = client.receive()
        print(
            "[Client][P4-debug] received OKVS setup: "
            f"num_docs={p4_setup.num_docs}, slots={p4_setup.table.size}, "
            f"value_size={p4_setup.table.value_size}, seed_prefix={p4_setup.table.seed[:8].hex()}"
        )
        p4_client = Protocol4Client(oprf_client=DHOPRFClient(params=DHOPRFParams()))
        p4_request = p4_client.make_query(query_tokens.cpu())
        print(f"[Client][P4-debug] sending OPRF blind request count={len(p4_request.elements)}")
        if p4_request.elements:
            print(f"[Client][P4-debug] first blinded element prefix={hex(p4_request.elements[0])[:34]}")
        client.send(p4_request)
        p4_response = client.receive()
        print(f"[Client][P4-debug] received OPRF response count={len(p4_response.elements)}")
        tf_recovered = p4_client.recover_term_frequencies(p4_response, p4_setup).to(DEVICE)
        print(f"[Client][P4-debug] recovered TF shape={tuple(tf_recovered.shape)}")
        print(f"[Client][P4-debug] recovered TF matrix:\n{tf_recovered}")

        df = (tf_recovered > 0).sum(dim=0).float()
        idf = torch.log1p((p4_setup.num_docs - df + 0.5) / (df + 0.5))
        weighted_tf_plain = idf.unsqueeze(0) * (RAG_CONFIG.bm25_k1 + 1.0) * tf_recovered
        print(f"[Client][P2-debug] df per query token={df.tolist()}")
        print(f"[Client][P2-debug] idf per query token={idf.tolist()}")
        print("[Client][P2-debug] document length terms stay server-side; client only shares TF-derived BM25 numerator.")
        print(f"[Client][P2-debug] weighted_tf numerator matrix:\n{weighted_tf_plain}")
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
        print(f"[Client][P2-debug] shared weighted_tf shape={s_weighted_tf_local[0].shape}")
        print(f"[Client][P2-debug] shared tf shape={s_tf_local[0].shape}")
        print(f"[Client][P2-debug] received length_norm share shape={length_norm_share.shape}")
        print(f"[Client][P2-debug] secure BM25 contribution share shape={lexical_contrib_share.shape}")
        print(f"[Client][P2-debug] secure BM25 score share shape={lexical_scores_share.shape}")

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

        semantic_indicators_server = client.receive()
        lexical_indicators_server = client.receive()
        semantic_indicators_plain = ArithmeticSecretSharing.restore_from_shares(
            semantic_indicators_server,
            semantic_result.indicators,
        ).convert_to_real_field().round().cpu()
        lexical_indicators_plain = ArithmeticSecretSharing.restore_from_shares(
            lexical_indicators_server,
            lexical_indicators,
        ).convert_to_real_field().round().cpu()
        print(f"[Client][Semantic-PIR-query] restored top-k indicators:\n{semantic_indicators_plain}")
        print(f"[Client][Lexical-PIR-query] restored top-k indicators:\n{lexical_indicators_plain}")
        client.send(semantic_indicators_plain)
        client.send(lexical_indicators_plain)

        semantic_client_share = client.receive()
        semantic_pir_audit = client.receive()
        lexical_client_share = client.receive()
        lexical_pir_audit = client.receive()
        my_doc_sem_share = ass_from_plain_share(semantic_client_share)
        my_doc_lex_share = ass_from_plain_share(lexical_client_share)
        print(f"[Client][P3-debug] masked semantic_scores shape={semantic_scores.shape}")
        print(f"[Client][P4-debug] lexical_scores_share shape={lexical_scores.shape}")
        print_topk_audit("Client", "Semantic", semantic_result.topk_audit)
        print_topk_audit("Client", "Lexical", lexical_topk.audit)
        print_pir_audit("Client", "Semantic", semantic_pir_audit)
        print_pir_audit("Client", "Lexical", lexical_pir_audit)

        # ---------------------------------------------------------
        # 5. 配合还原结果 (发送语义路的 Share 给 Server)
        # ---------------------------------------------------------
        #client.send(top_k_docs_sem_share)
        
        # ================== 【第二次假跑】 ==================
        print("[Client] 执行 Dummy Model 2 (Seq=56)...")
        dummy_ids_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_32 = torch.ones(1, TOTAL_SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_32, dummy_pos_32, dummy_typ_32, dummy_mask_32)
        # =========================================================        

        # # 1) 接收 Server 发来的【语义路】文档 Share
        # c_doc_sem_remote = client.receive()
        # my_doc_sem_share = c_doc_sem_remote[0] # [1, 24, V]
        
        # # 2) 接收 Server 发来的【词汇路(BM25)】文档 Share
        # c_doc_lex_remote = client.receive()
        # my_doc_lex_share = c_doc_lex_remote[0] # [1, 24, V]

        # # 3) 复用前面 Client 发 Query 时的 Share (长度 8)
        # my_query_share = s_ids[0][0]   

        # # 4) 密态无缝拼接 Query + SemDoc + LexDoc
        # print("[Client] 密态拼接 Query 和 双路 Document...")
        # my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)


        print("[融合] 使用 Suda PIR-to-share 输出的密态文档拼接真实 Token 序列...")

        # 3). 拿到 Client 发来的 Query Share，直接拼接！
        my_query_share = s_ids[0][0]
        my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_sem_share, my_doc_lex_share], dim=1)




        # 5) 构造 56 长度的 Pos, Typ, Mask
        joint_pos = torch.arange(TOTAL_SEQ).unsqueeze(0).to(DEVICE)
        
        # 核心：用 0 标识 Query，用 1 标识所有的 Document
        joint_typ = torch.cat([
            torch.zeros(1, QUERY_LEN),      # 前 8 个是 Query (Type 0)
            torch.ones(1, SEM_DOC_LEN),     # 中间 24 个是语义文档 (Type 1)
            torch.ones(1, LEX_DOC_LEN)      # 最后 24 个是 BM25文档 (Type 1)
        ], dim=1).long().to(DEVICE)
        
        joint_mask = torch.ones(1, TOTAL_SEQ, dtype=torch.float32).to(DEVICE)

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
        print("[Client] 执行联合推理...")
        seq_out, pool = model(my_joint_ids_share, my_pos_share, my_typ_share, RingTensor.convert_to_ring(joint_mask))
        
        # 6.还原结果
        client.send(pool)

    client.close()

if __name__ == "__main__":
    if os.environ.get("SKIP_GEN_PARAMS") == "1":
        print("=== [Init] SKIP_GEN_PARAMS=1，跳过辅助参数生成 ===\n")
    else:
        gen_params()
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=run_client)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print("\n[Done] RAG Baseline 执行完毕！")
