import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.signal

# def kernel_func(a,b, mode):
# 	if mode =="add":
# 		return a+b
# 	elif mode == "minus":
# 		return a-b

# def test(mode):
# 	def my_kernel_func(a,b):
# 		return kernel_func(a,b, mode)

# 	return my_kernel_func


# mm=test("minus")
# print(mm(1,2))

# a=[0,0,1,1]
# b=[0,1,0,1]

# plt.scatter(a,b, c= np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]]) )
# plt.show()

# a = np.array([[1,2],[3,4]])
# print(np.linalg.norm(a))

np.random.seed(1000)
x= np.random.normal(size=[20,20])
ker = np.random.normal(size= [3,3])
pool_ker = np.array([[1,1],[1,1]])/4

y1= scipy.signal.convolve2d (scipy.signal.convolve2d(x,ker), pool_ker)[0::2, 0::2].astype(np.int32)
y2 = scipy.signal.convolve2d (scipy.signal.convolve2d(x,pool_ker)[0::2,0::2], ker).astype(np.int32)

print(y1)
print(y2)