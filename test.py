import numpy as np

x_batch = np.array([[1,2],[3,4]])
w = np.array([1,2])
b = np.array([-10])
y_batch = np.array([2,2])


c=np.mean( np.expand_dims(np.maximum(1- y_batch*(np.matmul( x_batch,w )+b), 0)*y_batch, axis =-1) * x_batch, axis = 0 , keepdims = False)
d = np.mean(np.maximum(1- y_batch*(np.matmul( x_batch,w )+b), 0)*y_batch, axis =0, keepdims = False)

print(c.shape)
print(d)
# print(cc)