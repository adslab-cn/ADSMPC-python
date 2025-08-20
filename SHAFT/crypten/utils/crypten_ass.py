import torch
import crypten as ct
from crypten import communicator as comm

from NssMPC import RingTensor, ArithmeticSecretSharing

class CrypTenASS(ArithmeticSecretSharing):
    """
    一个继承自 NssMPClib.ArithmeticSecretSharing 的新类，
    但是其通信方法被重写为使用 CrypTen 的 communicator。
    """

    def __init__(self, ring_tensor):
        super().__init__(ring_tensor)
        self.comm = comm.get() # 获取全局的 CrypTen communicator
        if self.comm is None:
            raise RuntimeError("CrypTen communicator has not been initialized. Please call crypten.init() first.")

    def send(self, dst):
        """
        使用 CrypTen 的 isend 发送底层的 torch.Tensor 份额。
        
        :param dst: 目标参与方的 rank。
        """
        # 我们发送的是内部的 torch.Tensor，而不是整个对象
        share_tensor = self.ring_tensor.tensor
        return self.comm.isend(share_tensor, dst=dst)

    @classmethod
    def receive(cls, src, template_ass):
        """
        使用 CrypTen 的 irecv 接收一个 torch.Tensor 份额，然后重新包装成 CrypTenASS 对象。
        
        :param src: 发送方的 rank。
        :param template_ass: 一个同样类型的 ASS 对象，用于提供形状、类型等元数据。
        :return: 一个新的 CrypTenASS 对象。
        """
        communicator = comm.get()
        
        # 创建一个用于接收数据的占位符 tensor
        template_tensor = template_ass.ring_tensor.tensor
        placeholder = torch.empty_like(template_tensor)
        
        # 异步接收
        req = communicator.irecv(placeholder, src=src)
        req.wait() # 等待接收完成
        
        # 将接收到的 torch.Tensor 重新包装
        received_rt = RingTensor(placeholder, 
                                 dtype=template_ass.ring_tensor.dtype, 
                                 device=template_ass.ring_tensor.device)
        
        return cls(received_rt)

    def restore(self):
        """
        使用 CrypTen 的 all_reduce 来高效地重构秘密。
        这是最推荐的重构方式。
        
        :return: 重构后的 RingTensor (在所有参与方上都相同)。
        """
        share_tensor = self.ring_tensor.tensor
        
        # all_reduce 会在环上进行加法，这正是我们需要的
        # 注意：torch.distributed 的 all_reduce 默认是模 2^64 加法，
        # 这与 NssMPClib 的环 Z_L 语义可能需要对齐。
        # 假设它们是兼容的。
        reconstructed_tensor = self.comm.all_reduce(share_tensor)
        
        # 将重构后的 torch.Tensor 包装回 RingTensor
        return RingTensor(reconstructed_tensor, 
                          dtype=self.ring_tensor.dtype,
                          device=self.ring_tensor.device)

    # 还需要重写 share 方法，让它返回 CrypTenASS 实例
    @classmethod
    def share(cls, tensor: RingTensor, num_of_party: int = 2):
        """
        重写 share 方法，使其返回 CrypTenASS 对象的列表。
        """
        if num_of_party != comm.get().get_world_size():
            # 为了简化，我们假设参与方数量与 CrypTen 的 world_size 一致
            print(f"Warning: num_of_party ({num_of_party}) does not match CrypTen world_size ({comm.get().get_world_size()}).")

        # 使用父类的 share 逻辑，但用我们的 cls 来构造对象
        share = []
        x_0 = tensor.clone()

        for _ in range(num_of_party - 1):
            x_i = RingTensor.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(cls(x_i))
            x_0 -= x_i
        share.append(cls(x_0))
        return share