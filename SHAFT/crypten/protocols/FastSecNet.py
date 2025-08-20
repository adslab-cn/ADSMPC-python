import torch

from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.crypto.aux_parameter import Parameter
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey
from NssMPC.crypto.primitives import DCF
import crypten as ct
def convert_ass_to_mpctensor(ass_share: ArithmeticSecretSharing):
    """
    在每个参与方本地，将一个 NssMPClib 的 ArithmeticSecretSharing (ASS) 份额
    转换为一个 CrypTen 的 MPCTensor。

    这个过程是纯本地计算，不涉及任何通信。

    :param ass_share: 当前参与方持有的 NssMPClib 份额。
    :return: 一个代表相同秘密值的 CrypTen MPCTensor。
    """
    
    # 1. 从 ASS 对象中提取底层的 torch.Tensor 份额。
    #    ass_share.ring_tensor.tensor 包含了秘密共享的整数值。
    nss_share_tensor = ass_share.ring_tensor.tensor
    
    # 2. 获取 NssMPClib 份额的元数据。
    #    我们需要知道这个份额是代表整数还是定点数（浮点数）。
    nss_dtype = ass_share.ring_tensor.dtype # 应该是 'int' 或 'float'

    # 3. 使用 CrypTen 的 from_shares 方法进行“重新包装”。
    #    MPCTensor.from_shares 是专门为此类场景设计的。
    #    它告诉 CrypTen：“这里有一个现成的秘密共享份额，请直接用它来构建一个 MPCTensor”。
    from crypten.mpc.mpc import MPCTensor
    from crypten.mpc import ptype
    mpc_tensor = MPCTensor.from_shares(nss_share_tensor, ptype=ptype.arithmetic)
    
    # 4. 【关键步骤】恢复元数据：设置 CrypTen 张量的编码器。
    #    如果 NssMPClib 的份额是 'float' 类型，意味着它使用了定点数编码。
    #    我们也必须在 CrypTen 的 MPCTensor 上设置一个匹配的编码器，
    #    否则后续的解密或浮点运算会出错。
    if nss_dtype == 'float':
        # 假设 NssMPClib 和 CrypTen 使用了相同的缩放因子（例如 2**16）
        # 我们可以从 ass_share.ring_tensor.scale 获取确切的缩放因子
        scale = ass_share.ring_tensor.scale
        precision_bits = int(torch.log2(torch.tensor(scale, dtype=torch.float32)).item())
        
        from crypten.encoder import FixedPointEncoder
        mpc_tensor.encoder = FixedPointEncoder(precision_bits=precision_bits)
        
    return mpc_tensor
class FastSecNetReLUKey(Parameter):
    def __init__(self, dcf_key:DCFKey=None, r_ss=None, b_ss=None):
        self.dcf_key = dcf_key
        self.r_ss = r_ss
        self.b_ss = b_ss
    @staticmethod
    def gen(num_of_keys, alpha, device="cpu"):
        #b = RingTensor.stack([RingTensor.ones_like(alpha),alpha],dim=0)
        b = RingTensor.convert_to_ring(torch.Tensor([1, -1 * alpha.convert_to_real_field()[0].item()],device=device))
        dcf_key0,dcf_key1 =  DCFKey.gen(num_of_keys, alpha, -1 * b,device=device)
        R = ArithmeticSecretSharing.share(alpha)
        B = ArithmeticSecretSharing.share(b)
        return FastSecNetReLUKey(dcf_key0,R[0],B[0]),FastSecNetReLUKey(dcf_key1,R[1],B[1])
    

class FastSecNetReLU:
    @staticmethod
    def gen(num_of_keys, alpha, device="cpu"):
        return FastSecNetReLUKey.gen(num_of_keys, alpha, device)
    @staticmethod
    def eval(x_ss:ArithmeticSecretSharing, key:FastSecNetReLUKey, party_id):
        key.r_ss.to(x_ss.device)
        k_ss = x_ss + key.r_ss 
        #print("==="+str(k_ss))
        x_r =RingTensor(ct.communicator.get().all_reduce(k_ss.ring_tensor.tensor),dtype="float")
        #x_r = k_ss.restore()
        print(x_r.convert_to_real_field())
        res = ArithmeticSecretSharing(DCF.eval(x_r,key.dcf_key,party_id)) + key.b_ss
        res = convert_ass_to_mpctensor(res)
        x_r = x_r.convert_to_real_field()
        #print(x_r.convert_to_real_field())
        y = res[...,0] * x_r + res[...,1]
        return y
