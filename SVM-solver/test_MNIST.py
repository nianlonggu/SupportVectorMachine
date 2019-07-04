from svm import *
from utils import *
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x0= x_train[y_train == 4]
y0= np.ones( x0.shape[0] )

x1= x_train[y_train == 9]
y1= np.ones( x1.shape[0] ) *(-1)

x = np.concatenate( [ x0, x1 ], axis =0 )
x = np.reshape(x, [x.shape[0],-1] )/255
y = np.concatenate( [ y0, y1], axis =0 )

random_indx = np.random.permutation( np.arange( x.shape[0] ) )
x = x[random_indx]
y = y[random_indx]

x_val = x[:500]
y_val = y[:500]
x_test = x[500:1000]
y_test = y[500:1000]
x = x[1000:2000]
y = y[1000:2000]


sigma = np.mean( distance_matrix( x,y ) )*0.5

svm = SVM_Solver( kernel_type = {"name":"GAUSSIAN", "params":[sigma]} )
svm.train( x,y, x_val, y_val )
