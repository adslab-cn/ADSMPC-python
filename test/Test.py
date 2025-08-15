import threading
from NssMPC.application.neural_network.party.neural_network_party import NeuralNetworkCS
from NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples import AssMulTriples
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

# 从你的项目导入必要的模块
from NssMPC.config import NN_path, DEVICE
from NssMPC.config.runtime import PartyRuntime
from NssMPC.application.neural_network.utils.converter import load_model, share_model, share_data
from data.AlexNet.Alexnet import AlexNet # 确保这里的 Alexnet.py 已经被改回1通道输入

# --- 1. 初始化服务端和客户端 ---
# 使用你在 Test.py 中定义的 SemiHonestCS，这与教程中的 NeuralNetworkCS 角色相同
server = NeuralNetworkCS(type='server')
client = NeuralNetworkCS(type='client')

# 为乘法、比较等操作设置提供者 (provider)
server.set_multiplication_provider()
server.set_comparison_provider()
server.set_nonlinear_operation_provider()

client.set_multiplication_provider()
client.set_comparison_provider()
client.set_nonlinear_operation_provider()

# --- 2. 定义服务端和客户端的连接函数 ---
def set_server_online():
    """让服务端上线等待连接"""
    print("Server is going online...")
    server.append_provider(ParamProvider(param_type=AssMulTriples))
    server.online()
    print("Server is connected.")

def set_client_online():
    """让客户端上线并连接"""
    print("Client is going online...")
    client.append_provider(ParamProvider(param_type=AssMulTriples))
    client.online()
    print("Client is connected.")

# --- 3. 定义服务端的核心推理逻辑 ---
def server_predict():
    print("Server thread started.")
    with PartyRuntime(server):
        # 实例化一个为 MNIST (1通道) 设计的 AlexNet 模型结构
        net = AlexNet(in_channels=3)
    
        # 加载你刚刚正确训练好的、基于 MNIST 的模型文件
        # 注意：这里使用了正确的、绝对的路径！
        model_path = '/home/joker/.NssMPClib/data/NN/AlexNet_MNIST.pkl'
        print(f"Server loading model from: {model_path}")
        net.load_state_dict(torch.load(model_path))
        
        # 将模型参数进行秘密共享，并将其中一份发送给客户端
        shared_param, shared_param_for_other = share_model(net)
        server.send(shared_param_for_other)
        
        # 模拟一次推理流程以生成必要的辅助数据（如Beaver三元组）
        num = server.dummy_model(net)
        print(f"Server dummy run finished. Waiting for {num} inferences.")
        
        # 用自己持有的那份秘密份额加载模型
        net = load_model(net, shared_param)

        # 循环等待客户端的数据并进行安全推理
        while num > 0:
            print(f"Server waiting for data... ({num} inferences left)")
            shared_data = server.receive()
            server.inference(net, shared_data)
            num -= 1
            
    print("Server inference complete. Closing connection.")
    server.close()
    
# --- 4. 定义客户端的核心推理逻辑 ---
def client_predict():
    print("Client thread started.")
    with PartyRuntime(client):
        # 客户端也实例化一个同样的 AlexNet 模型结构
        net = AlexNet(in_channels=3)
        
        # 准备 MNIST 测试数据，这与训练和模型是匹配的！
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.MNIST(root=NN_path, train=False, download=True, transform=transform1)
        indices = list(range(5)) # 和教程一样，只取前5张图片测试
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False)
    
        # 接收来自服务端的模型参数秘密份额
        print("Client waiting for model share...")
        shared_param = client.receive()
        
        # 客户端也进行一次模拟推理
        num = client.dummy_model(test_loader)
        print(f"Client dummy run finished. Preparing to send {num} images.")
        
        # 加载自己那份模型
        net = load_model(net, shared_param)
        
        # 遍历测试数据
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
    
            # 将自己的数据进行秘密共享，并将其中一份发送给服务端
            print(f"Client sharing image {i+1}/5...")
            shared_data, shared_data_for_other = share_data(images)
            client.send(shared_data_for_other)
    
            # 执行安全推理，得到的结果也是秘密共享的形式
            print(f"Client performing inference on image {i+1}/5...")
            res = client.inference(net, shared_data)
    
            # 在PyTorch中，torch.max会返回最大值和对应的索引
            _, predicted = torch.max(res, 1)
    
            print(f"Image {i+1} - Actual label: {labels.item()}, Predicted result: {predicted.item()}")
    
    print("Client inference complete. Closing connection.")
    client.close()

# --- 5. 主程序入口：启动所有线程 ---
if __name__ == "__main__":
    # 首先，让双方建立网络连接
    AssMulTriples.gen_and_save(10000, saved_name='AssMulTriples', num_of_party=2, type_of_generation='TTP')
    server_conn_thread = threading.Thread(target=set_server_online)
    client_conn_thread = threading.Thread(target=set_client_online)
    
    server_conn_thread.start()
    client_conn_thread.start()
    
    server_conn_thread.join()
    client_conn_thread.join()
    
    print("\n--- Network connection established. Starting inference protocol. ---\n")
    
    # 连接建立后，再开始执行推理协议
    server_inference_thread = threading.Thread(target=server_predict)
    client_inference_thread = threading.Thread(target=client_predict)

    server_inference_thread.start()
    client_inference_thread.start()
    
    server_inference_thread.join()
    client_inference_thread.join()
    
    print("\n--- Inference finished successfully. ---")