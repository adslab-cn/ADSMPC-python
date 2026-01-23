#!/usr/bin/env python
# coding: utf-8

# # Tutorial 0: Before Starting
# If you want to know about the relevant information and usage instructions about this library, please refer to the README.md or the README.en.md.

# ## Auxiliary parameter generation
# Currently, we simulate the auxiliary parameters provided by a trusted third party using local files. If you want to generate these auxiliary parameters, please run```./debug/offline_parameter_generation.py```. The auxiliary parameters will be saved as a directory ```data``` by adding the base path .NssMPClib/data/ to the current user's home directory (Linux/Unix: /home/{username}; macOS: /Users/{username}; Windows: C:\Users\{username}). 
# 
# Additionally, you can change the number of parameters generated according to your needs. 
# In this way, the parameters can be generated as follows: 

# In[2]:


from NssMPC.config import VCMP_SPLIT_LEN
from NssMPC.crypto.aux_parameter import *
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.aux_parameter.select_keys.vos_key import VOSKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
# from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import MaliciousCMPKey, MACKey

gen_num = 100

AssMulTriples.gen_and_save(gen_num, saved_name='2PCBeaver', num_of_party=2, type_of_generation='TTP')
AssMulTriples.gen_and_save(gen_num, saved_name='3PCBeaver', num_of_party=3, type_of_generation='TTP')
BooleanTriples.gen_and_save(gen_num, num_of_party=2, type_of_generation='TTP')
RssMulTriples.gen_and_save(gen_num)

GrottoDICFKey.gen_and_save(gen_num)
DICFKey.gen_and_save(gen_num)
SigmaDICFKey.gen_and_save(gen_num)

GeLUKey.gen_and_save(gen_num)
TanhKey.gen_and_save(gen_num)
ReciprocalSqrtKey.gen_and_save(gen_num)
DivKey.gen_and_save(gen_num)

Wrap.gen_and_save(gen_num)
RssTruncAuxParams.gen_and_save(gen_num)

B2AKey.gen_and_save(gen_num)

MACKey.gen_and_save(gen_num)

VOSKey.gen_and_save(gen_num, 'VOSKey_0')
VOSKey.gen_and_save(gen_num, 'VOSKey_1')
VOSKey.gen_and_save(gen_num, 'VOSKey_2')

VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_0')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_1')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_2')

B2AKey.gen_and_save(gen_num, 'B2AKey_0')
B2AKey.gen_and_save(gen_num, 'B2AKey_1')
B2AKey.gen_and_save(gen_num, 'B2AKey_2')


# 1. **Beaver Triples** that will be used for multiplication for two-party arithmetic secret sharing, three-party arithmetic secret sharing, boolean secret sharing and replicated secret sharing.
# 2. Auxiliary parameters required by DICF(Distributed Interval Containment Function) comparison methods: keys used in the DICF method, which includes **DICFKey**, **GrottoDICFKey**, and **SigmaDICFKey**.
# 3. Auxiliary parameters associated with Look Up Table:
#    * **GeLU Key** is used for the Gaussian Error Linear Unit (GeLU) activation function.
#    * **Tanh Key** is associated with Tanh activation functions.
#    * **Reciprocal Sqrt Key** is used for dealing with square root reciprocal and negative index values.
#    * **Division Key** is used for division operation.
# 4. Auxiliary parameters associated with truncation operation:
#    * **Wrap Key** is used for truncation operation.
#    * **RSS Truncation Auxiliary Parameters** is used for truncation of RSS(Replicated Secret Sharing).
# 5. **B2A Key** implements conversion from boolean secret sharing to arithmetic secret sharing.
# 6. Auxiliary parameters associated with comparsion operation:
#    * **MAC Key** is the corresponding message authentication code key used for comparison operation verification.
#    * **Malicious CMP Key** implements malicious-secure comparison.
# 7. **VOSKey** is the auxiliary parameters required for the oblivious selection operation.
# 8. **VSigmaKey** is the FSS key class for verifiable sigma.

# In addition to these keys, other parameters are also required for some operations, such as the matrix beaver triples for matrix multiplication, which is related to the size of the matrix involved in the operation. Such parameters are usually related to actual operations, so it is hard to generate them in advance, and the generation scheme will be shown in subsequent tutorials.

# ## Configuration file
# Related configuration of the library will read the configuration file ```config.json```, which will be generated under the system path (Linux/Unix: /home/{username}; macOS: /Users/{username}; Windows: C:\Users\{username}). 
# Configuration files are used to define the operating parameters of applications. These parameters can include database connection information, network settings, user permissions, and so on. Therefore, users can easily adjust the behavior of the application.
# Now, let's have an insight into the basic configuration so that you can change the configuration to achieve different operations in the future. See the config section of the [API documentation](https://www.xidiannss.com/doc/NssMPClib/config.html) for more details.

# ## Some utils
# In ```./NssMPC/common/utils/debug_utils.py```, we provide some tools to help you debug the code. For example, you can use the following code to check the time of a function:

# In[ ]:


from NssMPC.common.utils import get_time
res = get_time(func, *args)


# `get_time` will return the result of the function and print the time it takes to run the function. The parameters of `get_time` are the function to be tested and the parameters of the function.
# 
# In addition, we provide a function for counting the communication costs in secure multiparty computation tasks. You can use the following code to check the communication cost of a function:

# In[ ]:


from NssMPC.common.utils import comm_count
res = comm_count(communicator, func, *args)


# `comm_count` will return the result of the function and print the communication cost of the function. The parameters of `comm_count` are the parties in `NssMPC/secure_model/mpc_party/semi_honest.py` or Communicator object in `NssMPC/common/network/communicator.py`, the function to be tested and the parameters of the function.

# #### Note
# If you find that some functions mentioned in this tutorial can not run, don't worry. It may be because the auxiliary parameters required for some functions are not generated or the auxiliary parameters are insufficient. You can refer to the tutorial and codes in the```./debug``` package to generate the auxiliary parameters required according to your needs, and distribute the calculation to multiple parties.
