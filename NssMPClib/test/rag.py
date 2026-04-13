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
DOC_LEN = 24
TOTAL_SEQ = QUERY_LEN + DOC_LEN

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
        
        # 【新增】：专门用来给假跑跑前向传播的明文模型，不参与实际密态推理
        model_for_dummy = SecBertModel(BERT_CONFIG)

        print("[Server] 执行 Dummy Model 1 (Seq=8)...")
        server.dummy_model(model_for_dummy)  # 【修改】：用明文模型假跑

        s_local, s_remote = share_model(model)



        server.send(s_remote)
        model = load_model(model, s_local)
        
        # ---------------------------------------------------------
        # 2. 准备服务端知识库 (Documents Database)
        # ---------------------------------------------------------
        print("[Server] 构建并分享密态知识库...")
        # 假设我们有 NUM_DOCS 篇文档，已经被编码为 hidden_size 维度的向量
        db_embeddings = torch.randn(NUM_DOCS, BERT_CONFIG['hidden_size']).to(DEVICE)
        
        # 服务端将数据库 Secret Share，一份留给自己，一份发给客户端
        s_db_local, s_db_remote = share_data(db_embeddings)
        server.send(s_db_remote)
        my_db_share = s_db_local[0] # 提取出 ArithmeticSecretSharing

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
        # 4. RAG 核心流程 (距离计算 -> 召回)
        # ---------------------------------------------------------
        print("[Server] RAG: 开始密态打分...")
        scores_share = secure_distance_computation_placeholder(query_emb_share, my_db_share)
        
        print("[Server] RAG: 开始密态 Top-K 召回...")
        top_k_docs_share = secure_top_k_retrieval_placeholder(scores_share, my_db_share, TOP_K)

        # ---------------------------------------------------------
        # 5. 还原结果进行验证
        # ---------------------------------------------------------
        c_top_k_docs = server.receive()
        final_docs = ArithmeticSecretSharing.restore_from_shares(top_k_docs_share, c_top_k_docs)
        print("\n=== [Server] RAG 最终召回的文档明文 (前两维) ===")
        print(final_docs.convert_to_real_field()[:, :2])
        
        print("[Server] 执行 Dummy Model 2 (Seq=32)...")
        server.dummy_model(model_for_dummy)
        
        # 1. 模拟将召回的一篇文档变成 Share (实际应用中是从 top_k_docs_share 提取)
        doc_ids = torch.tensor([[666, 777, 888, 999, 102] + [0]*(DOC_LEN-5)]).to(DEVICE)
        oh_doc_ids = F.one_hot(doc_ids, BERT_CONFIG['vocab_size']).float()
        s_doc_local, s_doc_remote = share_data(oh_doc_ids)
        server.send(s_doc_remote)
        my_doc_share = s_doc_local[0]  # [1, 24, V]

        # 2. 拿到 Client 之前发来的 Query Share
        # 注意：你需要确保在 Client 端发 Query 给 Server 时，Server 这里用 my_query_share 接住
        # my_query_share = sh_in[0][0] (利用前面 Server 收到的 Query 变量)
        my_query_share = sh_in[0]

        # 3. ★ 密态拼接 Query(长度8) 和 Document(长度24)
        print("[Server] 密态拼接 Query 和 Document...")
        my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_share], dim=1)

        # 4. 接收 Client 发来的辅助张量
        my_pos_share = server.receive()[0]
        my_typ_share = server.receive()[0]
        mask = server.receive()

        # 5. 执行联合推理
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
        print("[Client] RAG: 参与密态打分...")
        scores_share = secure_distance_computation_placeholder(query_emb_share, my_db_share)
        
        print("[Client] RAG: 参与密态 Top-K 召回...")
        top_k_docs_share = secure_top_k_retrieval_placeholder(scores_share, my_db_share, TOP_K)

        # ---------------------------------------------------------
        # 5. 配合还原结果
        # ---------------------------------------------------------
        client.send(top_k_docs_share)
        
        # ================== 【新增：第二次假跑】 ==================
        print("[Client] 执行 Dummy Model 2 (Seq=32)...")
        dummy_ids_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        dummy_pos_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        dummy_typ_32 = torch.zeros(1, TOTAL_SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)
        dummy_mask_32 = torch.ones(1, TOTAL_SEQ).to(DEVICE)
        client.dummy_model(dummy_ids_32, dummy_pos_32, dummy_typ_32, dummy_mask_32)
        # =========================================================        


        # 1. 接收 Server 发来的目标文档 Share
        c_doc_remote = client.receive()
        my_doc_share = c_doc_remote[0] # [1, 24, V]
        my_query_share = s_ids[0][0]   # 复用前面 Client 发 Query 时的 Share

        # 2. ★ 密态拼接 Query(长度8) 和 Document(长度24)
        print("[Client] 密态拼接 Query 和 Document...")
        my_joint_ids_share = ArithmeticSecretSharing.cat([my_query_share, my_doc_share], dim=1)

        # 3. 构造 32 长度的 Pos, Typ(0区分Query,1区分Doc), Mask
        joint_pos = torch.arange(TOTAL_SEQ).unsqueeze(0).to(DEVICE)
        joint_typ = torch.cat([torch.zeros(1, QUERY_LEN), torch.ones(1, DOC_LEN)], dim=1).long().to(DEVICE)
        joint_mask = torch.ones(1, TOTAL_SEQ, dtype=torch.float32).to(DEVICE)

        oh_joint_pos = F.one_hot(joint_pos, BERT_CONFIG['max_position_embeddings']).float()
        oh_joint_typ = F.one_hot(joint_typ, BERT_CONFIG['type_vocab_size']).float()

        # 4. 分享辅助张量并发送
        s_pos_local, s_pos_remote = share_data(oh_joint_pos); client.send(s_pos_remote)
        s_typ_local, s_typ_remote = share_data(oh_joint_typ); client.send(s_typ_remote)
        client.send(RingTensor.convert_to_ring(joint_mask))

        my_pos_share = s_pos_local[0]
        my_typ_share = s_typ_local[0]

        # 5. 执行联合推理并发送结果 Share
        print("[Client] 执行联合推理...")
        seq_out, pool = model(my_joint_ids_share, my_pos_share, my_typ_share, RingTensor.convert_to_ring(joint_mask))
        client.send(pool)

    client.close()

if __name__ == "__main__":
    gen_params()
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=run_client)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print("\n[Done] RAG Baseline 执行完毕！")