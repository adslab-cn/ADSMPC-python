#!/usr/bin/env python
# coding: utf-8

# # Tutorial 3: Replicated Secret Sharing
# Replicated Secret Sharing is used to securely share secret information among three parties, and in the scenario of three-parties replication, there are usually three participants work together to store and manage secrets. The secret is divided into three parts by some algorithm, with each participant holding one part. The separate parts don't have enough information to recover the secret. When a secret needs to be used, at least two participants need to cooperate, combining the parts they hold to recover the original secret. Currently, our models and features are based on honest-majority designs. To use replicated secret sharing in secure three-party calculations, we import the following packages

# In[1]:


from NssMPC import RingTensor
from NssMPC import ReplicatedSecretSharing
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import secure_ge
from NssMPC.secure_model.mpc_party import SemiHonest3PCParty, HonestMajorityParty
from NssMPC.config.runtime import PartyRuntime

import torch


# ```RingTensor``` is the main data structure that we use.
# ```HonestMajorityParty``` means participants in 3-party replicated secret sharing computation based on the honest majority model.
# ```ReplicatedSecretSharing``` is a class for replicated secret sharing over a RingPair.
# ```SemiHonest3PCParty``` is used for three parties to communicate with each other

# ## Party
# At least two parties are honest, meaning they will act according to the agreement and will not intentionally tamper with data or behavior. There may be a malicious party that tries to influence the outcome by providing incorrect information or tampering with data.

# In[2]:


import threading
Party0 = HonestMajorityParty(id=0)
Party1 = HonestMajorityParty(id=1)
Party2 = HonestMajorityParty(id=2)
# Set P0
def set_P0():
    with PartyRuntime(Party0):
        Party0.set_comparison_provider()
        # P0 connect
        Party0.online()

# Set P1
def set_P1():
    with PartyRuntime(Party1):
        Party1.set_comparison_provider()
        # P1 connect
        Party1.online()

# Set P2
def set_P2():
    with PartyRuntime(Party2):
        Party2.set_comparison_provider()
        # P2 connect
        Party2.online()
    
p0_thread = threading.Thread(target=set_P0)
p1_thread = threading.Thread(target=set_P1)
p2_thread = threading.Thread(target=set_P2)

p0_thread.start()
p1_thread.start()
p2_thread.start()
p0_thread.join()
p1_thread.join()
p2_thread.join()


# If you see three instances of "successfully connected" with two targets, it indicates that the communication between the three parties has been established successfully.

# ## Secret Sharing
# P0 will create 2 RingTensors( x and y）to test and use the ```share``` method from ```ReplicatedSecretSharing``` (RSS) to perform data sharing. Additionally, you need to utilize TCP to send party0's shares to another parties and receive their own shares.

# In[3]:


from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import secure_ge
from NssMPC.config.configs import DEVICE

def p0_share():
    with PartyRuntime(Party0):
        x = torch.tensor([1.1, 1.1, 1.3], device=DEVICE)
        y = torch.tensor([1.2, 1.1, 2.3], device=DEVICE)
        print(x)
        x = RingTensor.convert_to_ring(x)
        y = RingTensor.convert_to_ring(y)
        print("test_tensor: ", x)
        shares_X = ReplicatedSecretSharing.share(x)
        shares_Y = ReplicatedSecretSharing.share(y)
    
        Party0.send(1, shares_X[1])
        Party0.send(2, shares_X[2])
    
        Party0.send(1, shares_Y[1])
        Party0.send(2, shares_Y[2])
    
        shared_x = shares_X[0]
        shared_y = shares_Y[0]
        shared_x.party = Party0
        shared_y.party = Party0
        print(shared_x.restore().convert_to_real_field())
    
    
    
    
def p1_share():
    with PartyRuntime(Party1):
        shared_x = Party1.receive(0)
        shared_y = Party1.receive(0)
        print(shared_x.restore())
    

    
def p2_share():
    with PartyRuntime(Party2):
        shared_x = Party2.receive(0)
        shared_y = Party2.receive(0)
        print(shared_x.restore())
    
p0_thread = threading.Thread(target=p0_share)
p1_thread = threading.Thread(target=p1_share)
p2_thread = threading.Thread(target=p2_share)

p0_thread.start()
p1_thread.start()
p2_thread.start()
p0_thread.join()
p1_thread.join()
p2_thread.join()


# ####  Operations
# The operations in Replicated Secret Sharing are similiar to the operations in Arithmetic secret sharing.
