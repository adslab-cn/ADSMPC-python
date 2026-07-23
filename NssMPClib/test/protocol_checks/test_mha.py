import torch
import torch.nn as nn
import threading
import os
import sys

import torch.nn.functional as F
# --- 0. 环境设置 ---
# current_dir = os.getcwd() # 获取当前运行目录 (即 .../test)
# parent_dir = os.path.abspath(os.path.join(current_dir, "..")) # 获取上级目录 (即 .../NssMPClib)
# sys.path.append(parent_dir)
from NssMPC.config import DEVICE
from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.application.neural_network.layers.embedding import SecBertEmbeddings 
from NssMPC.application.neural_network.party.neural_network_party import NeuralNetworkCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.application.neural_network.utils.converter import share_model, load_model, share_data
from NssMPC.crypto.aux_parameter import AssMulTriples, MatmulTriples, GeLUKey,Wrap,SigmaDICFKey,GrottoDICFKey,DivKey,ReciprocalSqrtKey,TanhKey

# --- 这里的 import 路径对应你放置文件的位置 ---
from NssMPC.application.neural_network.layers.mha import SecBertSelfAttention,SecBertModel
def plaintext_inference():
    model = SecBertModel(BERT_CONFIG).to(DEVICE)
    model.eval()

    state_dict = torch.load("./NssMPClib/test/bert_tiny_weights.pth", map_location=DEVICE)
    
    # 1. 剥离 bert. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            new_state_dict[k[5:]] = v
        else:
            new_state_dict[k] = v
    new_state_dict.pop('embeddings.position_ids', None)

    # 2. 暴力替换参数，并强转为 float32 (解决 int64 截断归零问题)
    for name, param in model.named_parameters():
        if name in new_state_dict:
            param.data = new_state_dict[name].to(DEVICE).to(torch.float32)

    ids = torch.tensor([[101, 7592, 2088, 102] + [0]*(SEQ-4)]).to(DEVICE)
    pos = torch.arange(SEQ).unsqueeze(0).to(DEVICE)
    typ = torch.zeros_like(ids).to(DEVICE)
    mask = torch.ones_like(ids, dtype=torch.float32).to(DEVICE)

    oh_ids = F.one_hot(ids, BERT_CONFIG['vocab_size']).float()
    oh_pos = F.one_hot(pos, BERT_CONFIG['max_position_embeddings']).float()
    oh_typ = F.one_hot(typ, BERT_CONFIG['type_vocab_size']).float()

    with torch.no_grad():
        _, pool = model(oh_ids, oh_pos, oh_typ, mask)

    print("[Plaintext] pooler first 5:", pool[0, :5])
    return pool


BERT_CONFIG = {
    "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
    "intermediate_size": 512, "vocab_size": 30522, 
    "max_position_embeddings": 512, "type_vocab_size": 2
}
BATCH = 1
SEQ = 8 # 序列长

def gen_params():
    print("生成辅助参数...")
    if not os.path.exists('data'): os.makedirs('data')
    AssMulTriples.gen_and_save(1000000, saved_name='2PCBeaver')
    #MatmulTriples.gen_and_save(1000)
    DivKey.gen_and_save(10000)
    GeLUKey.gen_and_save(10000)
    TanhKey.gen_and_save(10000)
    Wrap.gen_and_save(10000)
    ReciprocalSqrtKey.gen_and_save(10000)
    SigmaDICFKey.gen_and_save(10000)
    print("参数生成完成")

# ==========================================
# 4. 执行逻辑
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
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        # 加载明文权重 (这部分逻辑不变)
        if os.path.exists("./NssMPClib/test/bert_tiny_weights.pth"):
            print("[Server] 加载 bert-tiny 权重...")
            state_dict = torch.load("./NssMPClib/test/bert_tiny_weights.pth", map_location=DEVICE)
            state_dict.pop('embeddings.position_ids', None)
            #model.load_state_dict(state_dict, strict=True)

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('bert.'):
                    new_state_dict[k[5:]] = v
                else:
                    new_state_dict[k] = v

            # ==================================================
            # 2. 暴力替换参数：直接覆盖 .data，拒绝 PyTorch 的强制取整！
            # ==================================================
            missing_keys = []
            for name, param in model.named_parameters():
                if name in new_state_dict:
                    # 强制把真实的浮点权重赋给模型，同时转换为 float64
                    param.data = new_state_dict[name].to(DEVICE).to(torch.float32)
                else:
                    missing_keys.append(name)
                    
            if missing_keys:
                print(f"\n[警告] 以下参数没有找到对应的权重: {missing_keys[:5]} ...\n")
            else:
                print("\n[Server] 权重暴力替换成功！")



            print(f"[Server] 权重加载完成")

            # 再次执行你的验证打印（此时 q_w 已经是 float64，mean() 不会再报错了）
            print("\n=== 检查 Layer 0 的 Q, K, V 权重 ===")
            layer_0_attn = model.encoder.layer[0].attention.self
            q_w = layer_0_attn.query.weight.data
            
            print(f"Q Weight | Shape: {q_w.shape} | Mean: {q_w.mean().item():.6f} | Max: {q_w.max().item():.6f}")
            print(f"\nQ Weight 局部切片 (前2行, 前5列):\n{q_w[:2, :5]}")



            print(f"[Server] 权重加载完成")
        else:
            print("[Server] 警告: 未找到权重文件，使用随机权重")

        # 执行 dummy_model ---
        print("[Server] 执行 Dummy Model...")
        server.dummy_model(model)

        # 再分享和发送模型
        s_local, s_remote = share_model(model)
        server.send(s_remote)

        # 加载密文权重
        model = load_model(model, s_local)
        
        # 等待输入并推理 (这部分逻辑不变)
        print("[Server] 等待输入...")
        sh_in = server.receive()
        sh_pos = server.receive()
        sh_type = server.receive()
        mask = server.receive()
        print("[Server] 进入推理")
        res, pool = model(sh_in[0][0], sh_pos[0][0], sh_type[0][0], mask)
        
        c_res = server.receive()
        final = ArithmeticSecretSharing.restore_from_shares(pool, c_res)
        mpc_out = final.convert_to_real_field()
        print("\n=== 最终结果 (Pooler Output 前5位) ===")
        print(mpc_out[0, :5])
        diff = torch.abs(plaintext_inference() - mpc_out)
        print("max diff:", diff.max())
        print("mean diff:", diff.mean())
                
    server.close()

def run_client():
    client.online()
    with PartyRuntime(client):
        model = SecBertModel(BERT_CONFIG)
        for param in model.parameters():
            param.requires_grad = False
        
        
        # 构造输入
        print("[Client] 构造输入...")
        ids = torch.tensor([[101, 7592, 2088, 102] + [0]*(SEQ-4)]).to(DEVICE)
        pos = torch.arange(SEQ).unsqueeze(0).to(DEVICE)
        typ = torch.zeros_like(ids).to(DEVICE)
        mask = torch.ones_like(ids, dtype=torch.float32).to(DEVICE)
        
        print("[Client] 转换 One-Hot 并加密...")
        oh_ids = F.one_hot(ids, BERT_CONFIG['vocab_size']).float()
        oh_pos = F.one_hot(pos, BERT_CONFIG['max_position_embeddings']).float()
        oh_typ = F.one_hot(typ, BERT_CONFIG['type_vocab_size']).float()


        #d_voc = torch.zeros(BATCH, SEQ, BERT_CONFIG['vocab_size']).to(DEVICE)
        #d_pos = torch.zeros(BATCH, SEQ, BERT_CONFIG['max_position_embeddings']).to(DEVICE)
        #d_typ = torch.zeros(BATCH, SEQ, BERT_CONFIG['type_vocab_size']).to(DEVICE)


        #执行 dummy_model ---
        print("[Client] 执行 Dummy Model...")
        #client.dummy_model(d_voc, d_pos, d_typ, mask)
        client.dummy_model(oh_ids,oh_pos,oh_typ,mask)

        # 再接收和加载模型
        s_local = client.receive()
        model = load_model(model, s_local)
        

        #分享输入
        s_ids = share_data(oh_ids); client.send(s_ids[1])
        s_pos = share_data(oh_pos); client.send(s_pos[1])
        s_typ = share_data(oh_typ); client.send(s_typ[1])
        client.send(RingTensor.convert_to_ring(mask))
        print("[Client] 进入推理")
        res, pool = model(s_ids[0][0], s_pos[0][0], s_typ[0][0], RingTensor.convert_to_ring(mask))
        
        client.send(pool)
    client.close()

if __name__ == "__main__":
    gen_params()
    t1 = threading.Thread(target=run_server)
    t2 = threading.Thread(target=run_client)
    t1.start(); t2.start()
    t1.join(); t2.join()