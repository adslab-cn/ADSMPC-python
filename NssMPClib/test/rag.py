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
TOP_K = 2      # 我们想要召回的文档数量
QUERY_LEN = 8
SEM_DOC_LEN = 24  # 语义路召回的文档长度
LEX_DOC_LEN = 24  # BM25路召回的文档长度
# 最终送入模型的总长度 = Query + Doc1(语义) + Doc2(词汇) = 56
TOTAL_SEQ = QUERY_LEN + SEM_DOC_LEN + LEX_DOC_LEN

def gen_params():
    print("=== [Init] 生成辅助参数 ===")
    if not os.path.exists('data'): os.makedirs('data')
    AssMulTriples.gen_and_save(1000000, saved_name='2PCBeaver')
    DivKey.gen_and_save(100000)
    GeLUKey.gen_and_save(100000)
    TanhKey.gen_and_save(100000)
    #MatmulTriples.gen_and_save(10000)
    Wrap.gen_and_save(100000)
    ReciprocalSqrtKey.gen_and_save(10000)
    SigmaDICFKey.gen_and_save(100000)
    B2AKey.gen_and_save(100000)
    print("=== [Init] 参数生成完成 ===\n")

# ==========================================
# 2. 核心 RAG 占位函数
# ==========================================
def secure_distance_computation_placeholder(query_emb_share, doc_embs_share):
    """
    [阶段 1] 密态距离计算
    query_emb_share: [1, 128]
    doc_embs_share: [NUM_DOCS, 128] (即 [10, 128])
    """
    # 使用逐元素相乘 (Element-wise Multiplication) 配合 PyTorch 广播机制
    # [1, 128] * [10, 128] -> [10, 128]
    # 这步会消耗普通的 AssMulTriples，不会触发 MatmulTriples 的形状报错
    element_wise_prod = query_emb_share * doc_embs_share
    
    # 沿着特征维度(dim=-1)求和，等价于计算内积
    # sum 会得到 [10] 维度的密态张量，这就是 10 篇文档的分数
    scores_share = element_wise_prod.sum(dim=-1)
    
    return scores_share

def secure_bm25_scoring_placeholder(client_keywords, server_okvs):
    """
    [阶段 1 - 词汇路] 密态 BM25 打分
    这里未来会跑 OPRF 和 OKVS 协议。
    返回: 针对所有文档的 BM25 分数的 Secret Share，形状 [1, NUM_DOCS]
    """
    # 假装算出了分数，用随机数代替
    fake_scores = torch.randn(1, NUM_DOCS).to(DEVICE)
    return ArithmeticSecretSharing(RingTensor.convert_to_ring(fake_scores))


def secure_top_k_retrieval_placeholder(scores_share, doc_embs_share, k):
    """
    [阶段 2] 密态 Top-K 召回：找出分数最高的 K 个文档。
    TODO: 这里是你未来要实现 Panther 协议 (ExactTopk/BatchPIR) 的地方！
    目前为了让代码跑通，我们直接返回前 K 个文档的切片（假装已经选出来了）。
    """
    # 丑陋的占位符：直接取前 K 篇文档的 Secret Share 返回
    # 实际上这里应该有一大堆的 secure_ge (密态比较) 或者 PIR 逻辑
    top_k_docs_share = doc_embs_share[:k] 
    return top_k_docs_share

# ==========================================
# 3. Server 与 Client 线程逻辑
# ==========================================
server = NeuralNetworkCS(type='server')
client = NeuralNetworkCS(type='client')
for p in [server, client]:
    p.set_multiplication_provider()
    p.set_comparison_provider()
    p.set_nonlinear_operation_provider()

def run_server():
    server.online()
    with PartyRuntime(server):
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
        db_embeddings = torch.randn(NUM_DOCS, BERT_CONFIG['hidden_size']).to(DEVICE)
        
        s_db_local, s_db_remote = share_data(db_embeddings)
        server.send(s_db_remote)
        my_db_share = s_db_local[0] 

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
        query_emb_share = pool # shape: [1, 128]
        
        # ---------------------------------------------------------
        # 4. RAG 核心流程：双路召回 (Dual-Path Retrieval)
        # ---------------------------------------------------------
        
        print("[Server] RAG: 开始双路密态打分与召回...")
        
        # 【第一路：语义检索 Semantic Path】
        scores_sem_share = secure_distance_computation_placeholder(query_emb_share, my_db_share)
        top_k_docs_sem_share = secure_top_k_retrieval_placeholder(scores_sem_share, my_db_share, k=1)

        # 【第二路：词汇检索 BM25 Path】
        server_okvs_mock = None # 随便给个 None 占位，防止报错
        client_keywords = ["apple", "price"] # 模拟客户端输入的词
        scores_lex_share = secure_bm25_scoring_placeholder(client_keywords, server_okvs_mock)
        top_k_docs_lex_share = secure_top_k_retrieval_placeholder(scores_lex_share, my_db_share, k=1)

        # ---------------------------------------------------------
        # 5. 还原结果进行验证 (这里我们验证一下语义路的结果)
        # ---------------------------------------------------------
        c_top_k_docs = server.receive()
        # 注意：这里用 top_k_docs_sem_share 还原
        final_docs = ArithmeticSecretSharing.restore_from_shares(top_k_docs_sem_share, c_top_k_docs)
        print("\n=== [Server] RAG 语义路召回的文档明文 (前两维) ===")
        print(final_docs.convert_to_real_field()[:, :2])
        
        print("[Server] 执行 Dummy Model 2 (Seq=56)...")
        server.dummy_model(model_for_dummy)
        
        print("[融合] 拼接 Semantic 和 BM25 召回的密态文档...")
        
        # 1). 模拟将【语义路】召回的文档转为密文 Share
        doc_sem_ids = torch.tensor([[666, 777, 888, 999, 102] + [0]*(SEM_DOC_LEN-5)]).to(DEVICE)
        oh_doc_sem = F.one_hot(doc_sem_ids, BERT_CONFIG['vocab_size']).float()
        s_doc_sem_local, s_doc_sem_remote = share_data(oh_doc_sem)
        server.send(s_doc_sem_remote)
        my_doc_sem_share = s_doc_sem_local[0]  # [1, 24, V]

        # 2). 模拟将【词汇路(BM25)】召回的文档转为密文 Share
        doc_lex_ids = torch.tensor([[111, 222, 333, 444, 102] + [0]*(LEX_DOC_LEN-5)]).to(DEVICE)
        oh_doc_lex = F.one_hot(doc_lex_ids, BERT_CONFIG['vocab_size']).float()
        s_doc_lex_local, s_doc_lex_remote = share_data(oh_doc_lex)
        server.send(s_doc_lex_remote)
        my_doc_lex_share = s_doc_lex_local[0]  # [1, 24, V]

        # 3). 拿到 Client 之前发来的 Query Share (长度 8)
        my_query_share = sh_in[0]

        # 4). 密态无缝拼接 Query + SemDoc + LexDoc
        print("[Server] 密态拼接 Query 和 双路 Document...")
        # 把三者在序列维度(dim=1)拼接，总长 8 + 24 + 24 = 56
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
        query_emb_share = pool

        # ---------------------------------------------------------
        # 4. RAG 核心流程 (参与距离计算 -> 参与召回)
        # ---------------------------------------------------------
        print("[Client] RAG: 参与双路密态打分与召回...")
        
        # 【第一路：语义检索 Semantic Path】
        scores_sem_share = secure_distance_computation_placeholder(query_emb_share, my_db_share)
        top_k_docs_sem_share = secure_top_k_retrieval_placeholder(scores_sem_share, my_db_share, k=1)

        # 【第二路：词汇检索 BM25 Path】
        server_okvs_mock = None 
        client_keywords = ["apple", "price"]
        scores_lex_share = secure_bm25_scoring_placeholder(client_keywords, server_okvs_mock)
        top_k_docs_lex_share = secure_top_k_retrieval_placeholder(scores_lex_share, my_db_share, k=1)

        # ---------------------------------------------------------
        # 5. 配合还原结果 (发送语义路的 Share 给 Server)
        # ---------------------------------------------------------
        client.send(top_k_docs_sem_share)
        
        # ================== 【第二次假跑】 ==================
        print("[Client] 执行 Dummy Model 2 (Seq=56)...")
        dummy_ids_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_32 = torch.ones(1, TOTAL_SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_32, dummy_pos_32, dummy_typ_32, dummy_mask_32)
        # =========================================================        

        # 1) 接收 Server 发来的【语义路】文档 Share
        c_doc_sem_remote = client.receive()
        my_doc_sem_share = c_doc_sem_remote[0] # [1, 24, V]
        
        # 2) 接收 Server 发来的【词汇路(BM25)】文档 Share
        c_doc_lex_remote = client.receive()
        my_doc_lex_share = c_doc_lex_remote[0] # [1, 24, V]

        # 3) 复用前面 Client 发 Query 时的 Share (长度 8)
        my_query_share = s_ids[0][0]   

        # 4) 密态无缝拼接 Query + SemDoc + LexDoc
        print("[Client] 密态拼接 Query 和 双路 Document...")
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
    gen_params()
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=run_client)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print("\n[Done] RAG Baseline 执行完毕！")