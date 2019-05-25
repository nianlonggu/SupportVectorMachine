import numpy as np
import matplotlib.pyplot as plt
import os

def generate_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

	return path


def load_data(num_samples = 1000, seed = 1000):

	np.random.seed(seed)
	x1 = np.random.multivariate_normal( [1,1], [[1,-0.3],[-0.3,2]], num_samples  )
	x2 = np.random.multivariate_normal( [7,7], [[1,-0.3],[-0.3,2]], num_samples )
	y1 = np.ones([num_samples]) *-1
	y2 = np.ones([num_samples]) *1
	x = np.concatenate([x1,x2], axis =0)
	y = np.concatenate([y1,y2], axis =0)

	return x, y


def plot_results( x,y, w=None, b=None, title = "", img_save_path = None , show_img = True ):

	x_negative_1 =[]
	x_positive_1 =[]
	for ind in range(x.shape[0]):
		if y[ind] == -1:
			x_negative_1.append( x[ind] )
		else:
			x_positive_1.append( x[ind] ) 
	x_negative_1 = np.asarray(x_negative_1)
	x_positive_1 = np.asarray(x_positive_1) 

	plt.plot(x_negative_1[:,0], x_negative_1[:,1], "o", markerfacecolor='none')
	plt.plot(x_positive_1[:,0], x_positive_1[:,1], 's', markerfacecolor='none')

	if w is not None and b is not None:
		if w[1] != 0:
			## a1 a2 represent the first and second dimensions
			a1 = np.linspace( min(x[:,0]), max(x[:,0]), 1000 )
			a2 = -w[0]/w[1]*a1-b/w[1]
			a2_up_margin = -w[0]/w[1]*a1-b/w[1]+1/w[1]
			a2_down_margin = -w[0]/w[1]*a1-b/w[1]-1/w[1]
			plt.plot(a1,a2,"r-")
			plt.plot(a1,a2_up_margin, "r--")
			plt.plot(a1,a2_down_margin, "r--")
		else:
			a2 = np.linspace( min(x[:,1]), max(x[:,1]), 1000 )
			a1 = -w[1]/w[0]*a2-b/w[0]
			a1_up_margin = -w[1]/w[0]*a2-b/w[0] + 1/w[0]
			a1_down_margin = -w[1]/w[0]*a2-b/w[0] - 1/w[0]
			plt.plot(a1,a2,"r-")
			plt.plot(a1_up_margin, a2, "r--")
			plt.plot(a1_down_margin, a2, "r--")

	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim([min(x[:,0])-0.5, max(x[:,0])+0.5 ])
	plt.ylim([min(x[:,1])-0.5, max(x[:,1])+0.5 ])
	plt.legend(["y=-1","y=+1"])
	plt.title(title)
	if img_save_path is not None:
		plt.savefig( img_save_path )
	if show_img:
		plt.show()

	plt.close()

## input the (x,y) of training dataset, output the hyperplane parameters w and b
def svm(x,y, max_iter = 100000, lr = 0.1, batch_size = 2000, mylambda = 0.0001 ):
	# initialize w and b
	x_dim = x.shape[-1]
	w = np.random.normal(size=x_dim)
	b = np.random.normal()
	current_iter = 0
	# start training
	while True:
		batch_index = np.random.choice( x.shape[0], size=batch_size ,replace = False )
		x_batch, y_batch = x[batch_index], y[batch_index]

		# define the loss function:
		loss1 = mylambda /2 * np.linalg.norm( w )**2
		loss2 = np.mean( np.maximum(1- y_batch*(np.matmul( x_batch,w )+b), 0) )
		loss = loss1 + loss2 
		dLdw = mylambda*w -  np.mean( np.expand_dims( ((1 - y_batch*(np.matmul( x_batch,w )+b)) >0)*y_batch, axis =-1) * x_batch, axis = 0 , keepdims = False)
		dLdb = - np.mean(((1- y_batch*(np.matmul( x_batch,w )+b))> 0)*y_batch, axis =0, keepdims = False)

		w -= lr * dLdw
		b -= lr * dLdb

		current_iter+=1
		if current_iter % 1000 ==0:
			print("regularization loss: %f, hinge loss: %f, totoal loss: %f"%( loss1,loss2, loss ))
			plot_results(x,y, w,b, title= "iteration %d"%(current_iter), img_save_path=generate_folder("results/gds-svm/")+"results-iter%d.jpg"%(current_iter), show_img=False )
		
		if current_iter >= max_iter:
			break
	return w,b


x, y = load_data()
np.random.seed()
w,b =svm(x,y)
plot_results(x,y, w,b)
