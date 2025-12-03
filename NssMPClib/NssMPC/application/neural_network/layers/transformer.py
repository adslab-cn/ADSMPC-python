# 文件: NssMPC/application/neural_network/layers/transformer.py

import torch.nn as nn
from NssMPC.application.neural_network.layers.linear import SecLinear
from NssMPC.application.neural_network.layers.normalization import SecLayerNorm

# 如果你的 layers/__init__.py 里没有暴露 SecLinear，请确保路径正确



class BertTransformerBlock(nn.Module):
    def __init__(self, n_heads, n_embd, attn_mask, qkv_format):
        super(BertTransformerBlock, self).__init__()
        
        self.n_heads = n_heads
        self.n_embd = n_embd
        
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        dim_W = n_embd // n_heads
        
        # 初始化子模块
        self.attn = MHADummy(n_heads, n_embd, dim_W, attn_mask, qkv_format, True)
        self.ffn = FFN(n_embd, 4 * n_embd)
        self.ln0 = SecLayerNorm(n_embd)
        self.ln1 = SecLayerNorm(n_embd)

    def forward(self, input_tensor):
        attn_out = self.attn(input_tensor)
        add0_out = attn_out + input_tensor
        ln0_out = self.ln0(add0_out)
        ffn_out = self.ffn(ln0_out)
        add1_out = ffn_out + ln0_out
        ln1_out = self.ln1(add1_out)
        
        return ln1_out