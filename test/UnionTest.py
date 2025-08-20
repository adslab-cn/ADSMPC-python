from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey
import torch

from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.crypto.primitives import DCF
from crypten.protocols.FastSecNet import FastSecNetReLU
from multiprocess_launcher import MultiProcessLauncher
import pickle
from crypten import communicator as comm
import crypten as ct
device = "cpu"
torch_plain = torch.tensor(
    [[3000000000000., 0.2, 3., -0.4, -5., -3000000., -6.4, 1.], [1940000000., 2., 3., -0.4, -5., 1., -6.5, 1.]],
    device=device)
plaintext_input = RingTensor(torch_plain)
num_of_keys = plaintext_input.numel()


X = ArithmeticSecretSharing.share(plaintext_input)
# 获取全局 communicator 实例
def UnionTest():
    device = "cpu"
    current_communicator = comm.get()
    my_rank = current_communicator.get_rank()
    x_encrypted = ct.cryptensor(torch_plain, ptype=ct.mpc.ptype.arithmetic)
    t = RingTensor([10.])
    t.tensor = x_encrypted.share
    t.dtype = "float"
    x_ss = ArithmeticSecretSharing(t)

    #x_ss = X[my_rank]
    #x_ss = ArithmeticSecretSharing(RingTensor.convert_to_ring(x_encrypted._tensor)).to(device)
    with open(f"k{my_rank}.pickle", 'rb') as f:
        key = pickle.load(f)
        key.dcf_key = DCFKey.from_dic(key.dcf_key)
    #print(key.r_ss)
    res = FastSecNetReLU.eval(x_ss, key, my_rank)
    res = res.get_plain_text()
    #reconstructed_result = current_communicator.all_reduce(res.ring_tensor.tensor)
    if my_rank == 0:
        #r.dtype = "float"
        #final_plaintext = RingTensor(reconstructed_result).convert_to_real_field()
        print("\nFinal plaintext result:")
        print(res)

        # 手动在明文中进行验证，以确保结果正确
        print("\n--- Verification ---")


if __name__ == "__main__":
    keys = FastSecNetReLU.gen(num_of_keys=16, alpha=RingTensor.convert_to_ring(torch.tensor([1000.])), device=device)
    with open("k0.pickle", 'wb') as f:
        pickle.dump(keys[0], f)
    with open("k1.pickle", 'wb') as f:
        pickle.dump(keys[1], f)
    x0 = ArithmeticSecretSharing(RingTensor([[ 6985089435896793799,  2492345874431554276, -5283640657976465879,
          7438385514942499011,  6043134930444583210, -1185694097246771309,
         -6110876124999185454, -2272867747985645164],
        [  752962382826213491,  4139471176031871644,   644269468864948578,
          6427208322282434472,  3302387657807493926, -1209550956737120568,
          3528427729819403416,  7631674423996246462]],dtype="float"))
    x1 = ArithmeticSecretSharing(RingTensor([[-6788481432341596871, -2492345874366005169,  5283640658042198487,
         -7438385514876989225, -6043134930379374890,  1185693900704307309,
          6110876125064302024,  2272867748051246700],
        [ -752835242920677491, -4139471175966204572,  -644269468799215970,
         -6427208322216924686, -3302387657742285606,  1209550956802722104,
         -3528427729754293400, -7631674423930644926]],dtype="float"))
    x = ArithmeticSecretSharing.restore_from_shares(x0,x1)
    print(x.convert_to_real_field())
    launcher = MultiProcessLauncher(2, UnionTest)
    launcher.start()
    launcher.join()
    launcher.terminate()