#!/usr/bin/env python
# coding: utf-8

#  # Tutorial 7: Neural Network Inference
# Our library can support inference of neural networks based on secret sharing. Here we present a tutorial of neural network inference using secure two-party computation and secure three-party computation. Similar to Tutorial_2, we simulate multiple parties using multi-threads and trusted third parties which provide auxiliary parameters using local files. Models are shared before the prediction, and data is shared during the prediction process. 
# 
# You can refer to `./debug/application/neural_network/2pc/neural_network_client.py` and `./debug/application/neural_network/2pc/neural_network_server.py` for examples of actual usage of the neural network in 2pc, refer to `./debug/application/neural_network/3pc/P0.py`, `./debug/application/neural_network/3pc/P1.py` and `./debug/application/neural_network/3pc/P2.py` for examples of actual usage of the neural network in 3pc.
#  In this tutorial, we use AlexNet as an example. First, train the model using `data/AlexNet/Alexnet_MNIST_train.py`. 

# In[ ]:


import torch
import torch.utils.data
import torchvision
from NssMPC.application.neural_network.utils.converter import load_model
from NssMPC.application.neural_network.utils.converter import share_model
from NssMPC.application.neural_network.utils.converter import share_data


# In[ ]:


# training AlexNet
# exec(open('..\data\AlexNet\CNN_MNIST_train.py').read())
exec(open('/home/adslab/code/ADSMPC-python/NssMPClib/data/AlexNet/Alexnet_MNIST_train.py').read())


# If there is a path error problem, it is because the code needs to download the model in advance.
# And then, import the following packages:

# In[ ]:


from data.AlexNet.Alexnet import AlexNet
from NssMPC.application.neural_network.party.neural_network_party import NeuralNetworkCS


# ### Now, we can create our two parties.

# With the server as the model provider and the client as the data provider, we need to generate triples for matrix multiplication in advance and distribute them to both parties. Similar to Tutorial_2, we simulate this process on the server side.
# The model provider also needs to import the following packages to share the model and data owner needs to import the following packages to share the data.

# In[ ]:


import threading

# set Server
server = NeuralNetworkCS(type='server')

def set_server():
    # CS connect
    
    # server.connect(server_server_address, server_client_address, client_server_address, client_client_address)
    server.online()
    
# set Client
client = NeuralNetworkCS(type='client')

def set_client():
    # CS connect
    
    # client.connect(client_server_address, client_client_address, server_server_address, server_client_address)
    client.online()
    
server_thread = threading.Thread(target=set_server)
client_thread = threading.Thread(target=set_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# The model provider needs to provide and share the model.The data provider needs to provide data. Because neural network inference involves matrix multiplication, before starting the prediction, we need to simulate one prediction and generate the required matrix Beaver triples ahead of time. The above steps are the preparation work. Before starting inference, the data provider needs to share its data. And then, the two parties load their respective model shares and start inference.

# In[ ]:


import torchvision.transforms as transforms
from torch.utils.data import Subset
from NssMPC.config import NN_path, DEVICE
from NssMPC.config.runtime import PartyRuntime
def server_predict():
    with PartyRuntime(server):
        net = AlexNet()
    
        net.load_state_dict(torch.load('/home/adslab/code/ADSMPC-python/NssMPClib/data/AlexNet/AlexNet_MNIST.pkl'))
        shared_param, shared_param_for_other = share_model(net)
        server.send(shared_param_for_other)
        
        num = server.dummy_model(net)
        net = load_model(net, shared_param)
        while num:
            shared_data = server.receive()
            server.inference(net, shared_data)
            num -= 1
    # close party after inference
    server.close()
    
def client_predict():
    with PartyRuntime(client):
        net = AlexNet()
        transform1 = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.MNIST(root=NN_path, train=False, download=True, transform=transform1)
        indices = list(range(5))
        subset_data = Subset(test_set, indices)
        test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)
    
        shared_param = client.receive()
        num = client.dummy_model(test_loader)
        net = load_model(net, shared_param)
        
        for data in test_loader:
            correct = 0
            total = 0
            images, labels = data
    
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
    
            shared_data, shared_data_for_other = share_data(images)
            client.send(shared_data_for_other)
    
            res = client.inference(net, shared_data)
    
            _, predicted = torch.max(res, 1)
    
            print("Predicted result is: ", predicted)
    
        # close party after inference
    client.close()
    
server_thread = threading.Thread(target=server_predict)
client_thread = threading.Thread(target=client_predict)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# P0 instantiates an AlexNet model and loads the weights of the pre-trained model, and P1 also instantiates an AlexNet model

# We can see the prediction results as above, the core statements used by our library for neural network prediction are `server.inference` and `client.inference`. If you wish to perform additional operations on the prediction results, you can process them according to your specific requirements.
# In [data/neural_network/AlexNet/Alexnet.py](https://gitee.com/xdnss/mpctensorlib/tree/master/data/neural_network/AlexNet/Alexnet.py) and [data/neural_network/ResNet/ResNet.py](https://gitee.com/xdnss/mpctensorlib/tree/master/data/neural_network/ResNet/ResNet.py), we provide the training code and pre-trained models for AlexNet and ResNet50. You can use them to train models according to your specific requirements and perform inference using these trained models.
