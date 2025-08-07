# import the libraries
from datetime import time

from NssMPC.crypto.aux_parameter import GeLUKey, SigmaDICFKey, B2AKey
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples import MatmulTriples
from NssMPC.config.runtime import PartyRuntime

import torch



import threading

from NssMPC.secure_model.utils.param_provider import ParamProvider

# set Server
# 1. 初始化两方计算环境 (服务器和客户端)
# 这个设置模仿了 Tutorial_2 中的多线程模拟方法
server = SemiHonestCS(type='server')
client = SemiHonestCS(type='client')


def setup_party(party):
    """一个通用函数来启动和连接一个计算方"""
    with PartyRuntime(party):
        party.append_provider(ParamProvider(param_type=GeLUKey))
        party.append_provider(ParamProvider(param_type=SigmaDICFKey))
        party.append_provider(ParamProvider(param_type=B2AKey))
        party.online()


# 在不同的线程中启动服务器和客户端
server_thread = threading.Thread(target=setup_party, args=(server,))
client_thread = threading.Thread(target=setup_party, args=(client,))
server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# 2. 定义 ReLU 计算函数
def sigma_relu_server(x_shift):
    """服务器端的计算逻辑"""
    with PartyRuntime(server):
        final_result = x_shift.restore().convert_to_real_field() #+ res_1
        print("final:"+str(final_result))
def sigma_relu_client(x_shift):
    with PartyRuntime(client):
        x_shift.restore()

if __name__ == "__main__":
    plaintext_input = torch.tensor([[30000000.,0.2,3.,-0.4,-5.,1.,-6.4,1.],[1.,2.,3.,-0.4,-5.,1.,-6.5,1.]])
    #plaintext_input = torch.randn(10, 5)
    #plaintext_input = torch.randn(52, 307)
    #plaintext_input = torch.tensor([[1,2,3,-4,-5,1,-6,1],[1,2,3,-4,-5,1,-6,1]])
    num_elements = plaintext_input.numel()
    GeLUKey.gen_and_save(num_elements)
    SigmaDICFKey.gen_and_save(num_elements)
    B2AKey.gen_and_save(num_elements)
    # 将 torch.tensor 转换为 RingTensor
    x_ring = RingTensor.convert_to_ring(plaintext_input)
    X = ArithmeticSecretSharing.share(x_ring, 2)

    server_relu_thread = threading.Thread(target=sigma_relu_server, args=(X[0],))
    client_relu_thread = threading.Thread(target=sigma_relu_client, args=(X[1],))

    server_relu_thread.start()
    client_relu_thread.start()
    client_relu_thread.join()
    server_relu_thread.join()


    server.close()
    client.close()