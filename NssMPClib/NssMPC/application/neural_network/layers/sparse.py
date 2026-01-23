"""
SecEmbedding is a custom embedding layer that extends PyTorch's nn.Module. It is primarily used for handling
one-hot encoded inputs and is used in secure computing environments (e.g., privacy-preserving machine learning). The
design of this layer makes it possible to safely share the embedding weights in multi-party computation.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share


class SecEmbedding(torch.nn.Module):
    """
    onehot operation have been done in the offline phase and on the plaintext only support the embedding layer at the beginning of the network.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        """
        :param num_embeddings: The size of the embedded dictionary, the number of different identifiers that can be embedded.
        :type num_embeddings: int
        :param embedding_dim: Each embedded dimension represents the size of the vector space that each identifier will be mapped to.
        :type embedding_dim: int
        :param padding_idx: Specified fill index
        :type padding_idx: int
        :param max_norm: the maximum norm constraint to the embedded vector.
        :type max_norm: int
        :param norm_type: The type of norm (default is **2.0**), indicates the use of the L2 norm.
        :type norm_type: float
        :param scale_grad_by_freq: The gradient update will be scaled according to the frequency.
        :type scale_grad_by_freq: bool
        :param sparse: Whether to use sparse updates.
        :type sparse: bool
        :param _weight: Optional initial embedding weights.
        :type _weight: torch.Tensor
        :param _freeze: Control whether the embedded weights are trainable (defaulting to **False**).
        :type _freeze: torch.Tensor
        :param device: The device where tensors are stored.
        :type device: str
        :param dtype: data type
        :type dtype: str
        """
        super(SecEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = torch.nn.Parameter(
                torch.zeros([num_embeddings, embedding_dim], dtype=torch.int64), requires_grad=False)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = torch.nn.Parameter(_weight, requires_grad=False)

        self.sparse = sparse


    def forward(self, x):
        # 1. 记录原始形状 [Batch, Seq, Vocab] -> [1, 8, 30522]
        original_shape = x.shape
        from NssMPC import RingTensor
        from NssMPC.crypto.primitives import ArithmeticSecretSharing
        # 2. 准备权重
        if isinstance(x, torch.Tensor):
            w = self.weight
        else:
            if isinstance(self.weight, (ArithmeticSecretSharing, RingTensor)):
                w = self.weight
            else:
                w = torch2share(self.weight, x.__class__, x.dtype)

        # ================= 【核心修改开始】 =================
        # 3. 降维打击：如果有 3 维 (Batch, Seq, Feature)，先展平成 2 维 (Batch*Seq, Feature)
        # 这样矩阵乘法和截断的结果就是 [8, 128]，与三元组形状完美匹配，不会报错
        is_3d_input = (len(original_shape) == 3)
        
        if is_3d_input:
            # 这里的 -1 会自动计算为 Batch * Seq (例如 1*8=8)
            # x 变成 [8, 30522]
            x = x.reshape(-1, original_shape[-1])

        # 4. 执行计算 (此时是 2D @ 2D)
        # z 的形状将是 [8, 128]
        z = x @ w

        # 5. 升维还原：变回 [Batch, Seq, Hidden] -> [1, 8, 128]
        # 这样后续的 LayerNorm 等层就能正常工作了
        if is_3d_input:
            # 获取 Embedding 维度 (Hidden Size)
            # 如果 w 是 SecretSharing，取 item.shape；如果是 Tensor 直接取 shape
            if hasattr(w, 'shape'):
                embed_dim = w.shape[1]
            else:
                # 兜底：从 self.weight 取，或者从 z.shape 推断
                embed_dim = self.embedding_dim 
            
            # z 是 SecretSharing 对象，支持 reshape
            z = z.reshape(original_shape[0], original_shape[1], -1)
        # ================= 【核心修改结束】 =================

        return z