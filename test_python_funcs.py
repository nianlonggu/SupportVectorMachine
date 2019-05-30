import numpy as np
import matplotlib.pyplot as plt



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

a=[0,0,1,1]
b=[0,1,0,1]

plt.scatter(a,b, c= np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]]) )
plt.show()