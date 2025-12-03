class FrontendBase():
    def __init__(self):
        self.backend = None

    def forward(self, x):
        self.backend(x)

    def set_backend(self, backend):
        self.backend = backend()

class MHADummy(FrontendBase):
    """
    对应 C++ 中的 _MHADummy
    """
    def __init__(self, n_heads, n_embd, dim_W, attn_mask, qkv_format, flag=True):
        super(MHADummy, self).__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        # 模拟 Attention 的输入输出维度变换
        self.dummy_layer = SecLinear(n_embd, n_embd)

    def forward(self, x):
        return self.dummy_layer(x)


class FFN(FrontendBase):
    def __init__(self, in_dim, hidden_dim):
        super(FFN, self).__init__()
        self.fc1 = SecLinear(in_dim, hidden_dim)
        self.fc2 = SecLinear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
