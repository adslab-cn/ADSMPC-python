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
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
            
        probs = self.softmax(scores)
        context = (probs @ v).permute(0, 2, 1, 3).contiguous()
        return context.reshape(*context.shape[:-2], self.hidden_size)


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
        return self.intermediate_act_fn(self.dense(hidden_states))

class SecBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SecLinear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = SecLayerNorm(config['hidden_size'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class SecBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SecBertAttention(config)
        self.intermediate = SecBertIntermediate(config)
        self.output = SecBertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
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
        party_type = "[Server]" if PartyRuntime.party.type == 'server' else "[Client]"
        print(f"{party_type} SecBertModel.forward: 开始")
        if attention_mask is not None:
            one = RingTensor.convert_to_ring(1.0).to(attention_mask.device)
            
            neg_val = RingTensor.convert_to_ring(-10000.0).to(attention_mask.device)
            
            extended_mask = (one - attention_mask) * neg_val
            
            extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None

        print(f"{party_type} SecBertModel.forward: Before Embeddings")
        embedding_output = self.embeddings(input_oh,  pos_oh, type_oh)
        print(f"{party_type} SecBertModel.forward: After Embeddings")

        print(f"{party_type} SecBertModel.forward: Before Encoder")
        sequence_output = self.encoder(embedding_output, extended_mask)
        print(f"{party_type} SecBertModel.forward: After Encoder")

        print(f"{party_type} SecBertModel.forward: Before Pooler")
        pooled_output = self.pooler(sequence_output)
        print(f"{party_type} SecBertModel.forward: After Pooler")
        
        print(f"{party_type} SecBertModel.forward: 结束")
        return sequence_output, pooled_output