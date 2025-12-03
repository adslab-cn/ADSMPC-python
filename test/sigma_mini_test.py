import os
import random

import torch
import time
import threading
from NssMPC import ArithmeticSecretSharing, RingTensor
from NssMPC.application.neural_network.layers.activation import  SecGELU
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC.config.runtime import PartyRuntime
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.function_secret_sharing.dicf import SigmaDICF
from NssMPC.crypto.aux_parameter import SigmaDICFKey, GeLUKey, B2AKey
from NssMPC.secure_model.utils.param_provider import ParamProvider

# 1. 初始化两方计算环境 (服务器和客户端)
# 这个设置模仿了 Tutorial_2 中的多线程模拟方法
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')


def setup_party(party):
    """一个通用函数来启动和连接一个计算方"""
    with PartyRuntime(party):
        party.append_provider(ParamProvider(param_type=GeLUKey))
        party.append_provider(ParamProvider(param_type=B2AKey))
        party.online()


# 在不同的线程中启动服务器和客户端



# 2. 定义 ReLU 计算函数
def fastsecnet_relu_server(x_shift):
    """服务器端的计算逻辑"""
    with PartyRuntime(server):
        start = time.time()
        relu_layer = SecGELU()
        res_0 = relu_layer(x_shift)
        spent_time = time.time() - start
        print("time for get secret share:" + str(spent_time))
        #res_1 = server.receive()

        final_result = res_0.restore().convert_to_real_field() #+ res_1
        print("final:"+str(final_result))
        spent_time = time.time() - start
        print("time for get plaintext:"+str(spent_time))

def fastsecnet_relu_client(x_shift):
    with PartyRuntime(client):
        relu_layer = SecGELU()
        res_1 = relu_layer(x_shift)
        #client.send(res_1)
        res_1.restore()

if __name__ == "__main__":
    #plaintext_input = torch.tensor([[30000000.,0.2,3.,-0.4,-5.,1.,-6.4,1.],[1.,2.,3.,-0.4,-5.,1.,-6.5,1.]])
    #plaintext_input = torch.randn(10, 5)g
    #plaintext_input = torch.randn(12, 3072)
    plaintext_input = torch.randn(52, 307)
    #plaintext_input = torch.tensor([[1,2,3,-4,-5,1,-6,1],[1,2,3,-4,-5,1,-6,1]])
    num_elements = plaintext_input.numel()
    GeLUKey.gen_and_save(num_elements)
    B2AKey.gen_and_save(num_elements)
    # 将 torch.tensor 转换为 RingTensor
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    X = ArithmeticSecretSharing.share(x_ring, 2)
    #r = RingTensor.random([1],down_bound=-3000,upper_bound=3000).to("float")
    r = RingTensor.convert_to_ring(torch.tensor([100000.]))
    #r = RingTensor.convert_to_ring(torch.tensor([3000000000.]))
    torch_gelu = torch.nn.GELU()
    plaintext_result = torch_gelu(plaintext_input).to(torch.float64).to("cuda")
    print("Expected PyTorch Result:\n", plaintext_result)
    server_relu_thread = threading.Thread(target=fastsecnet_relu_server, args=(X[0],))
    client_relu_thread = threading.Thread(target=fastsecnet_relu_client, args=(X[1],))

    server_thread = threading.Thread(target=setup_party, args=(server,))
    client_thread = threading.Thread(target=setup_party, args=(client,))
    server_thread.start()
    client_thread.start()
    client_thread.join()
    server_thread.join()
    server_relu_thread.start()
    client_relu_thread.start()
    client_relu_thread.join()
    server_relu_thread.join()


    server.close()
    client.close()