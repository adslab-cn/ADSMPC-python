import torch.nn as nn
from NssMPC.application.neural_network.layers.normalization import SecLayerNorm
from NssMPC.application.neural_network.layers.linear import SecLinear
from NssMPC.application.neural_network.layers.transformer import BertTransformerBlock


class BERT(nn.Module):
    
    """
    对应 C++ 中的 GPUBERT 主类
    """
    def __init__(self, n_layer, n_heads, n_embd, attn_mask=None, qkv_format=None):
        super(BERT, self).__init__()
        
        self.n_layer = n_layer
  
        self.blocks = nn.ModuleList([
            BertTransformerBlock(n_heads, n_embd, attn_mask, qkv_format)
            for _ in range(n_layer)
        ])

        self.ln_f = SecLayerNorm(n_embd)
        self.pool = SecLinear(n_embd, n_embd)

    def forward(self, x):
        x = self.ln_f(x)
            
        for block in self.blocks:
            x = block(x)
            
        return x