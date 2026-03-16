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
        from NssMPC.config.runtime import PartyRuntime
        import torch
        import torch.nn.functional as F

        if isinstance(x, torch.Tensor):
            normalized_shape_tuple = (self.normalized_shape,)
            weight_float = self.weight.to(x.dtype)
            bias_float = self.bias.to(x.dtype)
            return F.layer_norm(x, normalized_shape_tuple, weight_float, bias_float, self.eps)

        # --- 密文模式 ---
        else:
            # 判断当前身份，用于控制打印
            is_server = (PartyRuntime.party.type == 'server')

            def debug_print(tag, tensor_share):
                """
                内部辅助函数：双方同时调用 restore() 避免死锁，但只在 Server 端打印真实值
                """
                # 这一步包含了 send 和 receive，Client 和 Server 必须同时执行！
                restored = tensor_share.restore()
                if is_server:
                    print(f"\n--- [SecLayerNorm Debug] {tag} ---")
                    print(restored.convert_to_real_field())

            # 提前计算维度的浮点数倒数
            inv_shape = 1.0 / float(self.normalized_shape)

            # --- 1. 计算均值 ---
            sum_x = x.sum(dim=-1).unsqueeze(-1)
            mean = sum_x * inv_shape
            # debug_print("1. Mean", mean)

            # --- 2. 中心化 ---
            z = x - mean
            # debug_print("2. Centered Z (x - mean)", z)

            # --- 3. 计算方差 ---
            var = (z * z).sum(dim=-1).unsqueeze(-1) * inv_shape + self.eps
            # debug_print("3. Variance (var)", var)

            # --- 4. 计算标准差倒数 (极易发散的危险区) ---
            inv_sqrt_variance = x.__class__.rsqrt(var)
            # debug_print("4. Rsqrt (inv_sqrt_variance)", inv_sqrt_variance)
            from NssMPC import RingTensor
            # --- 5. 加载权重并广播 ---
            weight_ring = RingTensor(self.weight.data, dtype='float')
            weight_share = x.__class__(weight_ring)

            bias_ring = RingTensor(self.bias.data, dtype='float')
            bias_share = x.__class__(bias_ring)

            inv_sqrt_variance_bcast = mp_broadcast_to(inv_sqrt_variance, z.shape)
            weight_bcast = mp_broadcast_to(weight_share, z.shape)
            bias_bcast = mp_broadcast_to(bias_share, z.shape)

            # --- 6. 最终的缩放和平移 ---
            normalized_z = z * inv_sqrt_variance_bcast
            # debug_print("5. Normalized Z", normalized_z)

            scaled_z = normalized_z * weight_bcast
            final_z = scaled_z + bias_bcast
            # debug_print("6. Final Z", final_z)
            
            return final_z