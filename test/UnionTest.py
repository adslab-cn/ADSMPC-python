import torch

from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.crypto.primitives import DCF
from multiprocess_launcher import MultiProcessLauncher
import pickle
from crypten import communicator as comm

# 获取全局 communicator 实例
def UnionTest():
    device = "cuda"
    current_communicator = comm.get()
    my_rank = current_communicator.get_rank()
    plaintext_input = RingTensor(
        [[3000000000000., 0.2, 3., -0.4, -5., -3000000., -6.4, 1.], [1940000000., 2., 3., -0.4, -5., 1., -6.5, 1.]])
    num_of_keys = plaintext_input.numel()
    # alpha = RingTensor([50000.])
    # b = RingTensor.convert_to_ring(torch.Tensor([1, -1 * alpha.convert_to_real_field()[0].item()]))
    #dcf_key0, dcf_key1 = DCFKey.gen(num_of_keys, alpha, -1 * b)

    # with open('key0.pickle', 'wb') as f:
    #     pickle.dump(dcf_key0, f)
    # with open('key1.pickle', 'wb') as f:
    #     pickle.dump(dcf_key1, f)
    with open(f'key{my_rank}.pickle', 'rb') as f:
        dcf_key = pickle.load(f)
    with open(f'key0.pickle', 'rb') as f:
        dcf_key0 = pickle.load(f)
    with open(f'key1.pickle', 'rb') as f:
        dcf_key1 = pickle.load(f)
    # beta = -1 * b
    # Key0, Key1 = DCF.gen(num_of_keys=num_of_keys, alpha=alpha, beta=beta)

    # Party 0:
    res = DCF.eval(x=plaintext_input, keys=dcf_key, party_id=my_rank)
    # with open(f'res{my_rank}.pickle', 'wb') as f:
    #     pickle.dump(res, f)
    # with open(f'res{(my_rank+1)%2}.pickle', 'rb') as f:
    #     res_ = pickle.load(f)
    # temp = res + res_
    # temp.dtype = "float"
    #print(temp.convert_to_real_field())
    reconstructed_result = current_communicator.all_reduce(res.tensor)
    # res_0 = DCF.eval(x=plaintext_input, keys=dcf_key0, party_id=0)
    #
    # # Party 1:
    # res_1 = DCF.eval(x=plaintext_input, keys=dcf_key1, party_id=1)
    #
    # # restore result
    # res = res_0 + res_1
    # print(res)
    # temp = ArithmeticSecretSharing(res)
    # temp = temp.restore()
    # temp = temp.convert_to_real_field()
    # print(temp)
    # --- 4. 结果验证 (只在 rank 0 打印) ---
    if my_rank == 0:
        print("\n--- Reconstruction Complete ---")
        print("Reconstructed result (on ring):")
        print(reconstructed_result)

        # 如果需要，可以将 RingTensor 结果转换回浮点数进行验证
        # 假设您的 RingTensor 有一个 convert_to_real_field 方法
        r = RingTensor(reconstructed_result)
        #r.dtype = "float"
        final_plaintext = RingTensor(reconstructed_result).convert_to_real_field()
        print("\nFinal plaintext result:")
        print(final_plaintext)

        # 手动在明文中进行验证，以确保结果正确
        print("\n--- Verification ---")
        # 假设 alpha 和 beta 在生成密钥时是已知的
        # alpha_val = ...
        # beta_val = ...
        # expected_result = (plaintext_input_torch < alpha_val) * beta_val
        # print("Expected plaintext result:")
        # print(expected_result)
        # assert torch.allclose(final_plaintext, expected_result.float())
        # print("\nVerification successful!")


if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, UnionTest)
    launcher.start()
    launcher.join()
    launcher.terminate()