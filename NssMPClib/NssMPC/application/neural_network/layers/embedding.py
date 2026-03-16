#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
import torch.nn as nn
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
from NssMPC.application.neural_network.layers.normalization import SecLayerNorm

class SecEmbedding(torch.nn.Module):
    """
    Secure Embedding layer using Matrix Multiplication (One-Hot approach).
    """
    def __init__(self, num_embeddings=0, embedding_dim=0, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        super(SecEmbedding, self).__init__()
        
        if isinstance(num_embeddings, dict):
            num_embeddings = 0
            embedding_dim = 0
            
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        if _weight is None:
            # 只有当维度大于0时才创建 tensor，防止报错
            if num_embeddings > 0 and embedding_dim > 0:
                self.weight = torch.nn.Parameter(
                    torch.zeros([num_embeddings, embedding_dim], dtype=torch.float32)
                )
            else:
                # 占位符，load_model 后续会用真实的 Secret Share 权重覆盖它
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.weight = torch.nn.Parameter(_weight)

    # def forward(self, x):
    #     if isinstance(self.weight, (ArithmeticSecretSharing, RingTensor)):
    #         weight = self.weight
    #     else:
    #         weight = torch2share(self.weight, x.__class__, x.dtype)
    #     z = x @ weight
    #     return z
    def forward(self, x):
        original_shape = x.shape
        if isinstance(x, torch.Tensor):
            z = x @ self.weight
            #print("plain start"+"="*30)
            #print(x)
            #print(self.weight)
            #print(z)    
            #print("plain end"+"="*30)
        else:
            if isinstance(self.weight, (ArithmeticSecretSharing, RingTensor)):
                weight = self.weight

            else:
                weight = torch2share(self.weight, x.__class__, x.dtype)
            # 3. 动态压扁 (Flatten)
            # 将 [Batch, Seq, Vocab] -> [Batch*Seq, Vocab] (即 [8, 30522])
            # 这样做是为了匹配离线生成的 2D 三元组和截断密钥
            x_flat = x.reshape(-1, original_shape[-1])

            # 4. 执行计算 (此时是 2D @ 2D)
            # 结果 z_flat 是 [8, 128]
            # 此时形状与 WrapKey [8, 128] 完全一致，截断操作不会报错
            z_flat = x_flat @ weight

            # 5. 动态还原 (Unflatten)
            # 将结果变回 [Batch, Seq, Hidden] -> [1, 8, 128]
            # 获取当前 Embedding 的输出维度 (即 Hidden Size)
            hidden_size = z_flat.shape[-1]
            
            # 构造目标形状：原始形状的前 N-1 维 + 新的 Hidden 维
            target_shape = original_shape[:-1] + (hidden_size,)
            
            z = z_flat.reshape(*target_shape)
            #print("secret start"+"="*30)
            #print(x.restore().convert_to_real_field())
            #print(weight.restore().convert_to_real_field())
            #print(z.restore().convert_to_real_field())    
            #print("secret end"+"="*30)
            # ===============================================
            #z = x @ weight
        return z


class SecBertEmbeddings(torch.nn.Module):
    """
    Standard BERT Embeddings structure adapted for NssMPC.
    Accepts three separate one-hot tensors for words, positions, and token types.
    """
    def __init__(self, config):
        super(SecBertEmbeddings, self).__init__()
        self.word_embeddings = SecEmbedding(config['vocab_size'], config['hidden_size'])
        self.position_embeddings = SecEmbedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = SecEmbedding(config['type_vocab_size'], config['hidden_size'])

        self.LayerNorm = SecLayerNorm(config['hidden_size'])
        # 注意: Dropout 在安全计算中通常不使用，因为它引入了随机性。
        # 在推理时，Dropout层本身不起作用，所以保留它没有问题。
        self.dropout = torch.nn.Dropout(0.1) 

    def forward(self, input_oh, pos_oh, type_oh):
        # 1. 分别计算三种 embedding
        #    每个 one-hot 张量与对应的 embedding 权重矩阵相乘
        words_embeds = self.word_embeddings(input_oh)
        position_embeds = self.position_embeddings(pos_oh)
        token_type_embeds = self.token_type_embeddings(type_oh)
        #print("4")
        # 2. 将三种 embedding 相加
        #    这是 BERT embedding 的标准做法
        embeddings = words_embeds + position_embeds + token_type_embeds
        #print("5")
        # 3. 应用 Layer Normalization
        if isinstance(embeddings, torch.Tensor):
            print("Tensor plain before layer norm"+"="*30)
            print(embeddings)
        elif isinstance(embeddings, RingTensor):
            print("RingTensor plain before layer norm"+"="*30)
            print(embeddings.convert_to_real_field())
        elif isinstance(embeddings, ArithmeticSecretSharing):
            print("ArithmeticSecretSharing before layer norm"+"="*30)
            print(embeddings.restore().convert_to_real_field())
        embeddings = self.LayerNorm(embeddings)
        
        if isinstance(embeddings, torch.Tensor):
            print("Tensor plain after layer norm"+"="*30)
            print(embeddings)
        elif isinstance(embeddings, RingTensor):
            print("RingTensor plain after layer norm"+"="*30)
            print(embeddings.convert_to_real_field())
        elif isinstance(embeddings, ArithmeticSecretSharing):
            print("ArithmeticSecretSharing after layer norm"+"="*30)
            print(embeddings.restore().convert_to_real_field())
        #print("6")
        # 4. (可选) 应用 Dropout，在 eval() 模式下它什么也不做
        # embeddings = self.dropout(embeddings)
        #print("7")
        return embeddings