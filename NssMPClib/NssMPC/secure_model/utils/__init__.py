#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.


def mp_broadcast_to(source, target_shape):
    """
    通用 MPC 广播函数：将 source 张量广播到 target_shape。
    
    逻辑：
    1. 补齐维度：如果 source 维度少于 target，在前面补 1。
       例如：source [128], target [1, 8, 128] -> source 变为 [1, 1, 128]
    2. 扩展数据：使用 expand 将维度为 1 的地方撑大。
       例如：[1, 1, 128] -> [1, 8, 128]
    
    参数:
        source: ArithmeticSecretSharing, RingTensor 或 torch.Tensor
        target_shape: 目标形状 (tuple 或 torch.Size)
    """
    # 获取源形状
    if hasattr(source, 'shape'):
        src_shape = source.shape
    else:
        # 兜底，万一是个 list
        return source 

    # 1. 维度数量检查
    src_ndim = len(src_shape)
    tgt_ndim = len(target_shape)
    
    if src_ndim > tgt_ndim:
        raise ValueError(f"Source dimension ({src_ndim}) is larger than target dimension ({tgt_ndim}), cannot broadcast.")

    # 2. 构造中间形状 (Reshape View)
    # 如果源是 [128]，目标是 [B, S, 128] (3维)
    # 我们需要把源这就变成 [1, 1, 128]
    diff = tgt_ndim - src_ndim
    if diff > 0:
        # 前面补 diff 个 1
        new_view_shape = [1] * diff + list(src_shape)
        # 执行 reshape
        source = source.reshape(*new_view_shape)
    
    # 3. 执行扩展 (Expand)
    # 此时 source 已经是 [1, 1, 128]，target 是 [1, 8, 128]
    # 直接 expand 即可
    if source.shape != target_shape:
        source = source.expand(*target_shape)
        
    return source