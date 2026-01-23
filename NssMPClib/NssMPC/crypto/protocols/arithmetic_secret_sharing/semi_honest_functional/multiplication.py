#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

from NssMPC import RingTensor
from NssMPC.config.runtime import PartyRuntime
from NssMPC.crypto.aux_parameter import AssMulTriples, MatmulTriples
import torch

def beaver_mul(x, y):
    """
    Perform a Beaver's multiplication for secure multi-party computation (MPC) with ASS inputs.

    This function uses Beaver's multiplication triplets to securely compute the product of two ASS
    `x` and `y`. It handles ASS with different shapes by expanding the smaller ASS to match the larger one.

    :param x: The first ASS in the multiplication operation.
    :type x: ArithmeticSecretSharing
    :param y: The second ASS in the multiplication operation.
    :type y: ArithmeticSecretSharing
    :returns: The result of the Beaver's multiplication operation.
    :rtype: ArithmeticSecretSharing

    """
    try:
        broadcast_shape = torch.broadcast_shapes(x.shape, y.shape)
    except RuntimeError:
        # 如果无法广播，说明上层代码有问题
        raise ValueError(f"Shapes {x.shape} and {y.shape} are not broadcastable")

    # 2. 压扁成 2D
    # [1, 8, 128] -> [1024, 1]
    # 这样做是为了方便后续处理，并且和 1D 的三元组对齐
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    
    # 3. 如果需要，将 x 或 y 扩展到广播形状，再压扁
    # 比如 y 原本是 [1,8,1]，需要先 expand 成 [1,8,128]，再压扁成 [1024,1]
    if x.shape != broadcast_shape:
        x_flat = x.expand(broadcast_shape).reshape(-1, 1)
    if y.shape != broadcast_shape:
        y_flat = y.expand(broadcast_shape).reshape(-1, 1)

    # 4. 获取 1D 的 Beaver Triples
    # 数量 = 广播后形状的元素总数
    num_elements = 1
    for dim in broadcast_shape:
        num_elements *= dim
        
    party = PartyRuntime.party
    a, b, c = party.get_param(AssMulTriples, num_elements)
    a.dtype = b.dtype = c.dtype = x.dtype
    
    # 5. 把 1D 的三元组 Reshape 成 2D，以匹配压扁后的数据
    # a, b, c 现在都是 [1024, 1]
    a = a.reshape(x_flat.shape)
    b = b.reshape(y_flat.shape)
    c = c.reshape(x_flat.shape) # 结果形状跟随 x_flat

    # 6. 执行 Beaver 协议 (现在所有张量都是 2D 的，不会报错)
    e = x_flat - a
    f = y_flat - b

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = e_and_f.restore()
    common_e = common_e_f[:num_elements].reshape(x_flat.shape)
    common_f = common_e_f[num_elements:].reshape(y_flat.shape)

    res1 = RingTensor.mul(common_e, common_f) * party.party_id
    res2 = RingTensor.mul(a.item, common_f)
    res3 = RingTensor.mul(common_e, b.item)
    res_flat = res1 + res2 + res3 + c.item # 结果是 2D 的 [1024, 1]

    # 7. 还原原始形状
    # 把 [1024, 1] 变回 [1, 8, 128]
    res = x.__class__(res_flat.reshape(broadcast_shape))
    
    return res
    party = PartyRuntime.party
    a, b, c = party.get_param(AssMulTriples, x.numel())  # TODO: need fix, get triples based on x.shape and y.shape
    a.dtype = b.dtype = c.dtype = x.dtype
    a = a.reshape(x.shape)
    b = b.reshape(y.shape)
    c = c.reshape(x.shape)
    e = x - a
    f = y - b

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = e_and_f.restore()
    common_e = common_e_f[:x.numel()].reshape(x.shape)
    common_f = common_e_f[x.numel():].reshape(y.shape)

    res1 = RingTensor.mul(common_e, common_f) * party.party_id
    res2 = RingTensor.mul(a.item, common_f)
    res3 = RingTensor.mul(common_e, b.item)
    res = res1 + res2 + res3 + c.item

    res = x.__class__(res)
    return res


def secure_matmul(x, y):
    """

    Perform matrix multiplication with ASS inputs using beaver triples.

    This function uses Beaver's multiplication triplets to securely compute the matrix multiplication of two ASS
    `x` and `y`.

    :param x: The first ASS in the matrix multiplication operation.
    :type x: ArithmeticSecretSharing
    :param y: The second ASS in the matrix multiplication operation.
    :type y: ArithmeticSecretSharing
    :returns: The result of the matrix multiplication operation.
    :rtype: ArithmeticSecretSharing

    """
    party = PartyRuntime.party
    a_matrix, b_matrix, c_matrix = party.get_param(MatmulTriples, x.shape, y.shape)

    e = x - a_matrix
    f = y - b_matrix

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = e_and_f.restore()
    common_e = common_e_f[:x.numel()].reshape(x.shape)
    common_f = common_e_f[x.numel():].reshape(y.shape)

    res1 = RingTensor.matmul(common_e, common_f)
    res2 = RingTensor.matmul(common_e, b_matrix.item)
    res3 = RingTensor.matmul(a_matrix.item, common_f)

    res = res1 * party.party_id + res2 + res3 + c_matrix.item

    res = x.__class__(res)

    return res
