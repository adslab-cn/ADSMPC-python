import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import os
import sys

# --- 环境设置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.config import DEVICE
from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.application.neural_network.party.neural_network_party import NeuralNetworkCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.application.neural_network.utils.converter import share_model, load_model, share_data
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.crypto.aux_parameter import (
    AssMulTriples, MatmulTriples, GeLUKey, Wrap, SigmaDICFKey, GrottoDICFKey, 
    DivKey, ReciprocalSqrtKey, TanhKey
)
try:
    from NssMPC.crypto.aux_parameter import ExpKey
except ImportError:
    ExpKey = None

from NssMPC.application.neural_network.layers.linear import SecLinear
from NssMPC.application.neural_network.layers.activation import SecSoftmax, SecGELU, SecTanh
from NssMPC.application.neural_network.layers.normalization import SecLayerNorm
from NssMPC.application.neural_network.layers.embedding import SecBertEmbeddings 


class SecBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.query = SecLinear(self.hidden_size, self.hidden_size)
        self.key = SecLinear(self.hidden_size, self.hidden_size)
        self.value = SecLinear(self.hidden_size, self.hidden_size)
        self.softmax = SecSoftmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.head_dim)
        return x.reshape(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        is_secure_mode = isinstance(hidden_states, ArithmeticSecretSharing)
        
        party_type = ""
        if is_secure_mode:
            party_type = "[Server]" if PartyRuntime.party.type == 'server' else "[Client]"
        else:
            party_type = "[Plaintext]"
        debugging = False
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertSelfAttention.forward: 开始")
        
        q = self.transpose_for_scores(self.query(hidden_states))
        if debugging:
            if is_secure_mode:
                temp = q.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: q "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: q "+str(is_secure_mode)+"  "+str(q))

        k = self.transpose_for_scores(self.key(hidden_states))
        if debugging:
            if is_secure_mode:
                temp = k.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: k "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: k "+str(is_secure_mode)+"  "+str(k))
        
        v = self.transpose_for_scores(self.value(hidden_states))

        if debugging:
            if is_secure_mode:
                temp = v.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: v "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: v "+str(is_secure_mode)+"  "+str(v))
        
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        if debugging:
            if is_secure_mode:
                temp = scores.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: scores "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: scores "+str(is_secure_mode)+"  "+str(scores))
        probs = self.softmax(scores)
        if debugging:
            if is_secure_mode:
                temp = probs.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: probs "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: probs "+str(is_secure_mode)+"  "+str(probs))
            
        context = (probs @ v).permute(0, 2, 1, 3).contiguous()
        res = context.reshape(*context.shape[:-2], self.hidden_size)
        if debugging:
            if is_secure_mode:
                temp = res.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: res "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: res "+str(is_secure_mode)+"  "+str(res))
            if party_type != "[Client]":
                print(f"{party_type} SecBertSelfAttention.forward: 结束")
        return res

class SecBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SecLinear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = SecLayerNorm(config['hidden_size'])
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class SecBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SecBertSelfAttention(config)
        self.output = SecBertSelfOutput(config) 

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class SecBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SecLinear(config['hidden_size'], config['intermediate_size'])
        self.intermediate_act_fn = SecGELU()

    def forward(self, hidden_states):
        from NssMPC.config.runtime import PartyRuntime
        import torch
        debugging = False
        # --- 明文/Dummy 模式 ---
        if isinstance(hidden_states, torch.Tensor):
            dense_out = self.dense(hidden_states)
            act_out = self.intermediate_act_fn(dense_out)
            return act_out
            
        # --- 密文模式 ---
        else:
            is_server = (PartyRuntime.party.type == 'server')

            def debug_print(tag, tensor_share):
                """
                内部辅助函数：双方同时调用 restore() 避免死锁，但只在 Server 端打印真实值
                """
                # 这一步包含了 send 和 receive，Client 和 Server 必须同时执行！
                restored = tensor_share.restore()
                if is_server and debugging:
                    print(f"\n--- [SecBertIntermediate Debug] {tag} ---")
                    print(restored.convert_to_real_field())

            # 1. 打印输入
            debug_print("1. Input (hidden_states)", hidden_states)

            # 2. 执行 Dense 层并打印结果
            dense_out = self.dense(hidden_states)
            debug_print("2. After Dense (Linear)", dense_out)

            # 3. 执行 GeLU 激活并打印结果
            act_out = self.intermediate_act_fn(dense_out)
            debug_print("3. After GeLU (Output)", act_out)

            return act_out

class SecBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SecLinear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = SecLayerNorm(config['hidden_size'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)
def print_ass(tag,tensor):
    print("="*20)
    print("-"+str(tag))
    print(tensor.restore())
class SecBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SecBertAttention(config)
        self.intermediate = SecBertIntermediate(config)
        self.output = SecBertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        is_secure_mode = isinstance(hidden_states, ArithmeticSecretSharing)
        debugging = False
        
        party_type = ""
        if is_secure_mode:
            party_type = "[Server]" if PartyRuntime.party.type == 'server' else "[Client]"
        else:
            party_type = "[Plaintext]"
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertLayer.forward: 开始")
        attention_output = self.attention(hidden_states, attention_mask)
        if debugging:
            if is_secure_mode:
                temp = attention_output.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: attention_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: attention_output"+str(is_secure_mode)+"  "+str(attention_output))


        intermediate_output = self.intermediate(attention_output)
        if debugging:
            if is_secure_mode:
                temp = intermediate_output.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: intermediate_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: intermediate_output"+str(is_secure_mode)+"  "+str(intermediate_output))


        layer_output = self.output(intermediate_output, attention_output)
        if debugging:
            if is_secure_mode:
                temp = layer_output.restore()
                if party_type == "[Server]":
                    print(party_type+" is secure: layer_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: layer_output"+str(is_secure_mode)+"  "+str(layer_output))
            
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertLayer.forward: 结束")
        return layer_output

class SecBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([SecBertLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class SecBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SecLinear(config['hidden_size'], config['hidden_size'])
        self.activation = SecTanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        return self.activation(self.dense(first_token_tensor))
    
BERT_CONFIG = {
    "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
    "intermediate_size": 512, "vocab_size": 30522, 
    "max_position_embeddings": 512, "type_vocab_size": 2
}
class SecBertModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = BERT_CONFIG 
        
        self.config = config
        
        # 下面的代码保持不变
        self.embeddings = SecBertEmbeddings(config)
        self.encoder = SecBertEncoder(config)
        self.pooler = SecBertPooler(config)
    def forward(self, input_oh, pos_oh, type_oh, attention_mask=None):
        # party_type = "[Server]" if PartyRuntime.party.type == 'server' else "[Client]"
        # print(f"{party_type} SecBertModel.forward: 开始")
        # if attention_mask is not None:
        #     one = RingTensor.convert_to_ring(1.0).to(attention_mask.device)
            
        #     neg_val = RingTensor.convert_to_ring(-10000.0).to(attention_mask.device)
            
        #     extended_mask = (one - attention_mask) * neg_val
            
        #     extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
        # else:
        #     extended_mask = None
        is_secure_mode = isinstance(input_oh, ArithmeticSecretSharing)
        debugging = False
        
        party_type = ""
        if is_secure_mode:
            party_type = "[Server]" if PartyRuntime.party.type == 'server' else "[Client]"
        else:
            party_type = "[Plaintext]"
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: 开始")

        # --- 2. 根据模式处理 Mask ---
        extended_mask = None
        if attention_mask is not None:
            if is_secure_mode:
                # 【密文模式】: attention_mask 应该是 RingTensor
                # 确保 one 和 neg_val 也在正确的设备上
                one = RingTensor.convert_to_ring(1.0).to(attention_mask.device)
                neg_val = RingTensor.convert_to_ring(-10000.0).to(attention_mask.device)
                
                # RingTensor 运算
                extended_mask = (one - attention_mask) * neg_val
                extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
            else:
                # 【明文模式】: attention_mask 应该是 torch.Tensor
                # 直接用 PyTorch Tensor 运算
                extended_mask = (1.0 - attention_mask) * -10000.0
                extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)

        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: Before Embeddings")
        embedding_output = self.embeddings(input_oh,  pos_oh, type_oh)
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: After Embeddings")
        
        if debugging:
            if is_secure_mode:
                temp = embedding_output.restore()
                if party_type != "[Client]":
                    print(party_type+" is secure: embedding_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: embedding_output "+str(is_secure_mode)+"  "+str(embedding_output))

        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: Before Encoder")
        sequence_output = self.encoder(embedding_output, extended_mask)
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: After Encoder")

        if debugging:
            if is_secure_mode:
                temp = sequence_output.restore()
                if party_type != "[Client]":
                    print(party_type+" is secure: sequence_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: sequence_output "+str(is_secure_mode)+"  "+str(sequence_output))
        
        
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: Before Pooler")
        pooled_output = self.pooler(sequence_output)
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: After Pooler")

        if debugging:
            if is_secure_mode:
                temp = pooled_output.restore()
                if party_type != "[Client]":
                    print(party_type+" is secure: pooled_output "+str(is_secure_mode)+"  "+str(temp.convert_to_real_field()))
            else:
                print(party_type+" is secure: pooled_output "+str(is_secure_mode)+"  "+str(pooled_output))
        if party_type != "[Client]" and debugging:
            print(f"{party_type} SecBertModel.forward: 结束")
        
        
        return sequence_output, pooled_output