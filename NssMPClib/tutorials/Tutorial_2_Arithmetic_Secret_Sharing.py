#!/usr/bin/env python
# coding: utf-8

# # Tutorial 2: Arithmetic Secret Sharing
# Arithmetic secret sharing is mainly used in secure two-party computation, where each participant holds the shared value of the data. In this way the data does not leak information during the calculation process. At present, our model and functions are designed based on semi-honest parties.
# To use arithmetic secret sharing for secure two-party computation, we import the following packages

# In[ ]:


# import the libraries
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
from NssMPC import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.aux_parameter.beaver_triples.arithmetic_triples import MatmulTriples
from NssMPC.config.runtime import PartyRuntime

import torch


# ```SemiHonestCS``` is the two semi-honest party. ```ArithmeticSecretSharing``` is the main package that we use. ```RingTensor``` is the main data structure that we use. ```BeaverProvider``` is the triple provider we use in the arithmetic secret share for multiplication operations, and we use ```BeaverProvider``` to simulate a trusted third party to provide auxiliary operation data.

# ## Party
# First, we need to define the parties involved in the computation. For secure two-party computation, we need two parties: the server and the client.
# When setting up the parties, we need to specify the address and port for each party. Each party has a tcp server and a tcp client. They all need an address and a port. We also need to set the Beaver triple provider and the wrap provider for the computations. If you are planning to do comparison operations, do not forget to set the compare key provider.
# In this demonstration we are using multi-threading to simulate two parties. In real applications, the server and client run in two files. You can refer to ``./debug/crypto/primitives/arithmetic_secret_sharing/test_ass_server.py`` and ```./ debug/crypto/primitives/arithmetic_secret_sharing/test_ass_client.py```.

# In[ ]:


import threading

# set Server
server = SemiHonestCS(type='server')

server.set_multiplication_provider()
server.set_comparison_provider()
server.set_nonlinear_operation_provider()

def set_server():
    # CS connect
    server.online()

# set Client
client = SemiHonestCS(type='client')

client.set_multiplication_provider()
client.set_comparison_provider()
client.set_nonlinear_operation_provider()

def set_client():
    # CS connect
    client.online()


server_thread = threading.Thread(target=set_server)
client_thread = threading.Thread(target=set_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# If you see two instances of "successfully connected", it indicates that the communication between the two parties has been established successfully.

# ## Secret Sharing
# If both parties have data that they want to compute on without revealing their individual data to each other, you can use the ```share``` method from ```ArithmeticSecretSharing``` (ASS) to perform data sharing. Additionally, you need to utilize TCP to send each party's shares to the other party and receive their own shares.
# In this case, let's assume that the server has data denoted as x, and the client has data denoted as y.

# In[ ]:


from NssMPC.config.configs import DEVICE

# data belong to server
x = RingTensor.convert_to_ring(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE))
# data belong to client
y = RingTensor.convert_to_ring(torch.tensor([[-1.0, 2.0], [4.0, 3.0]], device=DEVICE))

# split x into 2 parts
X = ArithmeticSecretSharing.share(x, 2)

# split y into 2 parts
Y = ArithmeticSecretSharing.share(y, 2)

temp_shared_x0=X[0]
temp_shared_x1=X[1]
temp_shared_y0=Y[0]
temp_shared_y1=Y[1]

print(X[0], X[1])
print(Y[0], Y[1])


# In[ ]:


def server_action():
    with PartyRuntime(server):
        # server shares x1 to client
        server.send(X[1])
        shared_x_0 = ArithmeticSecretSharing(X[0].ring_tensor)
        # server receives y0 from client
        y0 = server.receive()
        shared_y_0 = ArithmeticSecretSharing(y0.ring_tensor)
        print("shared x in server: ", shared_x_0)
        print("\n")
        print("shared y in server: ", shared_y_0)

def client_action():
    with PartyRuntime(client):
        # client receives x1 from server
        x1 = client.receive()
        # client shares y0 to server
        client.send(Y[0])
        shared_x_1 = ArithmeticSecretSharing(x1.ring_tensor)
        shared_y_1 = ArithmeticSecretSharing(Y[1].ring_tensor)
        print("shared x in client: ", shared_x_1)
        print("\n")
        print("shared y in client: ", shared_y_1)

server_thread = threading.Thread(target=server_action)
client_thread = threading.Thread(target=client_action)



# In[ ]:


server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# ## Secret Restoring
# If you want to restore the original value by the share, you can use the ```restore()``` method, which returns a ```RingTensor``` value, and then the ```convert_to_real_field``` can recover the result.
# In this tutorial, we only print the recovered results on the server side.

# In[ ]:


# restore share_x
# server

print("temp_shared_x0",temp_shared_x0)
def restore_server():
    with PartyRuntime(server):
        restored_x = temp_shared_x0.restore()
        real_x = restored_x.convert_to_real_field()
        print("\n x after restoring:", real_x)

# client
def restore_client():
    with PartyRuntime(client):
        temp_shared_x1.restore()

server_thread = threading.Thread(target=restore_server)
client_thread = threading.Thread(target=restore_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# ## Operations
# Next, we'll show you how to use arithmetic secret sharing to achieve secure two-party computation.

# #### Arithmetic Operations

# In[ ]:


# Addition
# restore result
def addition_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 + temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\nAddition", result_restored)

def addition_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 + temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=addition_server)
client_thread = threading.Thread(target=addition_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# In[ ]:


# Subtraction
# restore result
def subtraction_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 - temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\nSubtraction", result_restored)

def subtraction_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 - temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=subtraction_server)
client_thread = threading.Thread(target=subtraction_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# In[ ]:


# Multiplication
# restore result
def multiplication_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 * temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\n Multiplication", result_restored)

def multiplication_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 * temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=multiplication_server)
client_thread = threading.Thread(target=multiplication_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# Note: Since all the beaver triples used were generated during the offline phase, don't forget to generate the required matrix beaver triples before performing matrix multiplication.

# In[ ]:


# Matrix Multiplication
from NssMPC.config.configs import DEBUG_LEVEL

def server_matrix_multiplication():
    with PartyRuntime(server):
        # gen beaver triples in advance
        if DEBUG_LEVEL != 2:
            triples = MatmulTriples.gen(1, x.shape, y.shape)
            server.providers[MatmulTriples].param = [triples[0]]
            server.send(triples[1])
            server.providers[MatmulTriples].load_mat_beaver()
    
        print('x @ y: ', x @ y)
        print('real_field(x @ y): ',(x @ y).convert_to_real_field())
        share_z = temp_shared_x0 @ temp_shared_y0
        res_share_z = share_z.restore().convert_to_real_field()
        print('restored x @ y: ', res_share_z)
        assert torch.allclose((x @ y).convert_to_real_field(), res_share_z, atol=1e-3, rtol=1e-3) == True

def client_matrix_multiplication():
    with PartyRuntime(client):
        if DEBUG_LEVEL != 2:
            client.providers[MatmulTriples].param = [client.receive()]
            client.providers[MatmulTriples].load_mat_beaver()
    
        share_z = temp_shared_x1 @ temp_shared_y1
        print('restored x @ y: ', share_z.restore().convert_to_real_field())


server_thread = threading.Thread(target=server_matrix_multiplication)
client_thread = threading.Thread(target=client_matrix_multiplication)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# #### Comparison Operations
# The output results ```0``` and ```1``` correspond to the ``False`` and ``True`` values obtained from comparing the sizes of the RingTensors.

# In[ ]:


# Server less than
def less_than_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 < temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\n(x < y)", result_restored)
    
# Client less than
def less_than_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 < temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=less_than_server)
client_thread = threading.Thread(target=less_than_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# In[ ]:


# Less than or equal
def less_equal_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 <= temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\n(x <= y)", result_restored)

def less_equal_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 <= temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=less_equal_server)
client_thread = threading.Thread(target=less_equal_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# In[ ]:


# Greater than
def greater_than_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 > temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\n(x > y)", result_restored)

def greater_than_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 > temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=greater_than_server)
client_thread = threading.Thread(target=greater_than_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()


# In[ ]:


# Greater than or equal
def greater_equal_server():
    with PartyRuntime(server):
        res_0 = temp_shared_x0 >= temp_shared_y0
        result_restored = res_0.restore().convert_to_real_field()
        print("\n(x >= y)", result_restored)

def greater_equal_client():
    with PartyRuntime(client):
        res_1 = temp_shared_x1 >= temp_shared_y1
        res_1.restore()

server_thread = threading.Thread(target=greater_equal_server)
client_thread = threading.Thread(target=greater_equal_client)

server_thread.start()
client_thread.start()
client_thread.join()
server_thread.join()

