#!/usr/bin/env python
# coding: utf-8

# # Tutorial 5: Parameter & Provider
# Now let's introduce an important module of our library: Parameter.
# 
# Parameter is like an abstract class, and all classes that inherit it and override __init__ and gen methods in a particular format will be automatically managed using the parameter provider. 
# 
# For example, the beaver triplet, which are generated in the offline phase and used in the online phase, should inherit from the Parameter class.
# So that parameter provider can load these pregenerated instances of the class which inherits from Parameter into the memory on demand and take them out when needed.

# ## How to create and use a new type of Parameter
# We can follow the following steps:
# 
# First, create a new class that inherits the Parameter class. 
# 

# In[1]:


from NssMPC import RingTensor, ArithmeticSecretSharing

from NssMPC.crypto.aux_parameter import Parameter


class MyParameter(Parameter):
    def __init__(self, a=None, b=None, c=None):
        # What attributes does the parameter contain
        self.attr_a = a
        self.attr_b = b
        self.attr_c = c
    
    @staticmethod
    def gen(num, param0):
        # By what process is this parameter produced
        # An example
        a = RingTensor.random((num,))
        b = RingTensor.convert_to_ring(param0).repeat((num,))
        c = a + b
        a_0, a_1 = ArithmeticSecretSharing.share(a)
        b_0, b_1 = ArithmeticSecretSharing.share(b)
        c_0, c_1 = ArithmeticSecretSharing.share(c)
        return MyParameter(a_0, b_0, c_0), MyParameter(a_1, b_1, c_1)  # A pair of instances should be returned 


# Then we can generate the required number of class instances.
# 
# You can also use the second parameter to specify the name of the saved file, but if you change the default Settings, you need to do so later when initializing the provider.Otherwise, this parameter defaults.

# In[2]:


MyParameter.gen_and_save(100, 'testParam', 72)


# When you have completed this step, you can call the following method to retrieve the required number of auxiliary parameters for calculation.

# In[ ]:


from NssMPC.secure_model.utils.param_provider import ParamProvider
from NssMPC.secure_model.mpc_party import SemiHonestCS

p = SemiHonestCS('client')
p.append_provider(ParamProvider(MyParameter, 'testParam'))
p.online()

my_param = p.get_param(MyParameter, 3)
# do some calculate using my_param


# The server side also needs to be set up this way, you can consider encapsulating this process into a custom Party class.
