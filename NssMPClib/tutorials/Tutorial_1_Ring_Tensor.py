#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: Ring Tensor
# In NssMPClib, the basic data structure is ```RingTensor```, which means the tensor on the ring, corresponding to the tensor of ```torch```. We transform the tensor of ```torch``` to the ```RingTensor``` and perform operations on it, which supports multiple data types of ```torch``` (int64, int32, float64, float32).
# Now let's start by importing the RingTensor package.

# In[1]:


from NssMPC import RingTensor
import torch


# ### Conversion between torch tensors and RingTensor
# The lib provides a ``convert_to_ring`` method to convert a tensor from ``torch.Tensor`` to ``RingTensor``. The ```convert_to_real_field``` method converts a ``RingTensor`` data to a ``torch.Tensor`` type.

# In[2]:


# Create torch tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)

# Convert a tensor to ring
x_on_ring = RingTensor.convert_to_ring(x)
print(x_on_ring)

# Convert a RingTensor to real field
x_real_field = x_on_ring.convert_to_real_field()
print(x_real_field)


# This lib supports the ` ` ` torch.int64 ` ` ` , ` ` ` torch.int32 ` ` ` , ` ` ` torch.float64 ` ` ` and ` ` ` torch.float32 ` ` ` type of data conversion to the ring.

# In[3]:


# Convert different data type tensor to ring
# torch.int64
x_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
print(x_int64)

x_int64_on_ring = RingTensor.convert_to_ring(x_int64)
print(x_int64_on_ring)

# torch.int32
x_int32 = torch.tensor([1, 2, 3], dtype=torch.int32)

x_int32_on_ring = RingTensor.convert_to_ring(x_int32)
print(x_int32_on_ring)

# torch.float64
x_float64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

x_float64_on_ring = RingTensor.convert_to_ring(x_float64)
print(x_float64_on_ring)

# torch.float32
x_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

x_float32_on_ring = RingTensor.convert_to_ring(x_float32)
print(x_float32_on_ring)


# ## Operations on Ring tensor
# Now let's look at what we can do with our ```RingTensor```
# 
# #### Arithmetic Operations
# We can carry out regular arithmetic operations between ```RingTensors```. These operations return a ring tensor output.

# In[4]:


# Arithmetic operations between RingTensors
x_on_ring = RingTensor.convert_to_ring(torch.tensor([1.0, 2.0, 3.0]))

y_on_ring = RingTensor.convert_to_ring(torch.tensor([2.0]))


# Addition
res_on_ring = x_on_ring + y_on_ring
print("\nAddition:", res_on_ring.convert_to_real_field())

# Subtraction
res_on_ring = x_on_ring - y_on_ring
print("\nSubtraction", res_on_ring.convert_to_real_field())

# Multiplication
res_on_ring = x_on_ring * y_on_ring
print("\nMultiplication", res_on_ring.convert_to_real_field())

# Matrix Multiplication
y_on_ring = RingTensor.convert_to_ring(torch.tensor([[1.0], [2.0], [3.0]]))
res_on_ring = x_on_ring @ y_on_ring
print("\nMatrix Multiplication", res_on_ring.convert_to_real_field())


# #### Comparisons
# Similarly, we can compute element-wise comparisons on ```RingTensors```. Different from arithmetic operations, comparisons performed on ```RingTensors``` will return ```True``` or ```False```, which is like comparisons between ```torch``` tensors.

# In[5]:


#Comparisons between RingTensors
x_on_ring = RingTensor.convert_to_ring(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

y_on_ring = RingTensor.convert_to_ring(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]))

# Less than
result = x_on_ring < y_on_ring
print("\n(x < y) :", result)

# Less than or equal
result = x_on_ring <= y_on_ring
print("\n(x <= y) :", result)

# Greater than
result = x_on_ring > y_on_ring
print("\n(x > y) :", result)

# Greater than or equal
result = x_on_ring >= y_on_ring
print("\n(x >= y) :", result)

# Equal
result = x_on_ring == y_on_ring
print("\n(x == y) :", result)

# Not Equal
result = x_on_ring != y_on_ring
print("\n(x != y) :", result)


# #### Other operations
# The library also supports some other operations on RingTensor, such as reshape, stack, cat, unsqueeze and so on.
# 

# In[6]:


x_on_ring = RingTensor.convert_to_ring(torch.tensor([1.0, 2.0, 3.0]))
y_on_ring = RingTensor.convert_to_ring(torch.tensor([4.0, 5.0, 6.0]))

# Concatenation
res_on_ring = RingTensor.cat((x_on_ring, y_on_ring))
print("Concatenation: \n", res_on_ring.convert_to_real_field())

# Stacking
res_on_ring = RingTensor.stack((x_on_ring, y_on_ring))
print("\nConcatenation: \n", res_on_ring.convert_to_real_field())

# Reshaping
res_on_ring_after_reshape = res_on_ring.reshape(-1, 6)
print("\nReshape: \n", res_on_ring_after_reshape.convert_to_real_field())

# UnSqueezing
res_on_ring = x_on_ring.unsqueeze(dim=1)
print("\nUnSqueezing: \n", res_on_ring.convert_to_real_field())


# Some functions like where, random, arange is also supported in the RingTensor, but the usage is a little different from above. 

# In[7]:


from NssMPC.common.ring.ring_tensor import RingTensor

# Where
x_on_ring = RingTensor.convert_to_ring(torch.tensor([1.0, 5.0, 3.0]))
y_on_ring = RingTensor.convert_to_ring(torch.tensor([4.0, 2.0, 6.0]))
condition = x_on_ring > y_on_ring

res_on_ring = RingTensor.where(condition, x_on_ring, y_on_ring)
print("\nWhere: \n", res_on_ring.convert_to_real_field())

# Random
res_on_ring = RingTensor.random(shape=(2, 3), dtype='int', device='cpu', down_bound=0, upper_bound=10)
print("\nRandom: \n", res_on_ring.convert_to_real_field())

# Arange
res_on_ring = RingTensor.arange(start=0, end=10, step=2, dtype='int', device='cpu')
print("\nArange: \n", res_on_ring.convert_to_real_field())


# Note that the condition of the where function must be a RingTensor, and the rest of the arguments can be RingTensor or int.
