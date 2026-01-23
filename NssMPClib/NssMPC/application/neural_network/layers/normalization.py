"""
The SecLayerNorm class is used to implement layer normalization.
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import torch
from NssMPC.application.neural_network.functional.functional import torch2share
from NssMPC.config import data_type


class SecLayerNorm(torch.nn.Module):
    """
    * The implementation of this class is mainly based on the `paper Sigma <https://eprint.iacr.org/2023/1269.pdf>`_.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        """
        Initialize the SecLayerNorm class.

        :param normalized_shape: The dimension that needs normalization is usually the last dimension of the input tensor.
        :type normalized_shape: int
        :param eps: Constants are used to prevent division by zero errors (default is **1e-05**).
        :type eps: float
        :param elementwise_affine: The parameter indicates whether to use learnable scaling and translation (weight and bias) (the default value set to **True**).
        :type elementwise_affine: bool

        ATTRIBUTES:
            * **normalized_shape** (*int*): The dimension that needs normalization is usually the last dimension of the input tensor.
            * **eps** (*float*): Constants are used to prevent division by zero errors (default is **1e-05**).
            * **weight** (*torch.Tensor*): The weights of neural networks.
            * **bias** (*torch.Tensor*): bias term
            * **scale** (*torch.Tensor*): the scale of the tensor.
            * **zero_point** (*int*): The weights of neural networks.
            * **elementwise_affine** (*torch.Tensor*): The weights of neural networks.

        """
        super(SecLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones([normalized_shape], dtype=data_type), requires_grad=False)
        self.bias = torch.nn.Parameter(
            torch.zeros([normalized_shape], dtype=data_type), requires_grad=False)
        self.scale = None
        self.zero_point = None
        self.elementwise_affine = elementwise_affine

    # def forward(self, x):
    #     """
    #     The forward propagation process.

    #     Start by summing the input ``x`` along the last dimension, divided by the mean obtained by ``self.normalized_shape``.
    #     The input data is then centralized, i.e. the mean is subtracted from each element. After calculating the
    #     reciprocal of the square root of variance ``inv_sqrt_variance``, we use the :func:`~NssMPC.application.neural_network.functional.functional.torch2share` function to obtain weight
    #     and bias. Finally, we multiply the centralized data by the reciprocal of the standard deviation and apply
    #     scaling (if ``elementwise_affine`` is enabled). And then you add the ``bias``.

    #     :param x: Input tensor, typically coming from the output of the previous layer.
    #     :type x: ArithmeticSecretSharing or ReplicatedSecretSharing
    #     :return: A tensor after layer normalization
    #     :rtype: ArithmeticSecretSharing or ReplicatedSecretSharing
    #     """
    #     mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape

    #     z = x - mean
    #     inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)

    #     weight = torch2share(self.weight, x.__class__, x.dtype)
    #     bias = torch2share(self.bias, x.__class__, x.dtype)

    #     z = z * inv_sqrt_variance * weight + bias

    #     return z
    def forward(self, x):
        """
        The forward propagation process with dual logic for dummy and secure modes.
        """
        from NssMPC.secure_model.utils import mp_broadcast_to
        # 导入必要的模块
        import torch
        import torch.nn.functional as F
        if isinstance(x, torch.Tensor):
            normalized_shape_tuple = (self.normalized_shape,)
            weight_float = self.weight.to(x.dtype)
            bias_float = self.bias.to(x.dtype)
            return F.layer_norm(x, normalized_shape_tuple, weight_float, bias_float, self.eps)

        else:
            # print("11")
            # mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape
            # print("12")
            # z = x - mean
            # print("13")
            # inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)
            # print("14")
            # weight = torch2share(self.weight, x.__class__, x.dtype)
            # print("15")
            # bias = torch2share(self.bias, x.__class__, x.dtype)
            # print("16")
            # z = z * inv_sqrt_variance * weight + bias
            # print("17")
            # return z
            mean = x.sum(dim=-1).unsqueeze(-1) / self.normalized_shape
            
            # 2. 中心化 [Batch, Seq, Hidden]
            z = x - mean
            
            # 3. 计算方差倒数平方根 [Batch, Seq, 1]
            inv_sqrt_variance = x.__class__.rsqrt(((z * z).sum(dim=-1).unsqueeze(-1)) / self.normalized_shape)
            
            # ================== 【核心修改开始】 ==================
            # 问题：inv_sqrt_variance 是 [1, 8, 1]，而 z 是 [1, 8, 128]
            # 密文乘法不支持自动广播，必须手动 expand
            inv_sqrt_variance = mp_broadcast_to(inv_sqrt_variance, z.shape)
            
            # 4. 获取权重
            weight = torch2share(self.weight, x.__class__, x.dtype)
            bias = torch2share(self.bias, x.__class__, x.dtype)
            
            # 权重也要处理广播 (上次我们处理过，这里再加强一下)
            # weight 原本是 [128]，需要变成 [1, 1, 128] 然后 expand 到 [1, 8, 128]
            # 或者是直接利用 RingTensor 的 expand
            if weight.shape != z.shape:
                # 先把维度数量对齐 (例如从 [128] -> [1, 1, 128])
                view_shape = [1] * (len(z.shape) - 1) + [-1]
                weight = weight.reshape(view_shape)
                bias = bias.reshape(view_shape)
                # 再撑大到和 z 一样
                weight = weight.expand(z.shape)
                bias = bias.expand(z.shape)
            # ================== 【核心修改结束】 ==================

            # 5. 执行计算
            # 现在所有参与运算的变量都是 [1, 8, 128] 了
            # beaver_mul 里的 b.reshape(y.shape) 就变成了 b.reshape([1, 8, 128])
            # 1024 对 1024，完美匹配！
            z = z * inv_sqrt_variance * weight + bias
            
            return z