import numpy as np
import matplotlib.pyplot as plt
import os

def generate_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

	return path


def load_data(num_samples = 1000):

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
def svm(x,y, max_iter = 100000, lr = 0.00001,  c1=10,  epsilon= 1e-5, plot_training_results = False ):
	# initialize w and b
	x_dim = x.shape[-1]
	w = np.random.normal(size=x_dim)
	b = np.random.normal()

	num_samples = x.shape[0]

	## we also need to initialize the dual parameter lamb and perform gradient descent algorithm on it
	lamb = np.absolute(np.random.normal(size = x.shape[0] ))
	current_iter = 0
	# start training
	while True:
		dLdlamb = -1 +  np.matmul(np.matmul( np.expand_dims(y , axis =-1)*x , x.T), y*lamb)  + c1*np.dot(lamb, y)*y 
		lamb -= lr * dLdlamb
		## perform clip
		lamb = np.maximum(lamb, 0)
		current_iter+=1
		if current_iter % 20000 ==0:
			# define the loss function:
			loss1 = -1*np.sum( lamb ) + 0.5*np.linalg.norm( np.matmul(x.T, lamb*y ) )**2 
			loss2 = c1 * 0.5 * np.dot(lamb, y) **2
			loss = loss1  +loss2
			print("loss1: %f, loss2: %f, overall loss: %f"%( loss1, loss2, loss ))
			w = np.matmul( x.T, lamb*y )
			# select a support vector
			support_ind = []
			for ind in range(x.shape[0]):
				if lamb[ind] > epsilon:
					support_ind.append(ind)
			x_support, y_support, lamb_support = x[support_ind], y[support_ind], lamb[support_ind]
			w = np.matmul( x_support.T, lamb_support*y_support )
			b = np.mean( y_support - np.matmul( x_support,w ))
			if plot_training_results:
				plot_results(x,y, w,b, title= "iteration %d"%(current_iter), img_save_path=generate_folder("results/gds-dual-svm/")+"results-iter%d.jpg"%(current_iter), show_img=False )
		
		if current_iter >= max_iter:
			break
	# select a support vector
	support_ind = []
	for ind in range(x.shape[0]):
		if lamb[ind] > epsilon:
			support_ind.append(ind)
	x_support, y_support, lamb_support = x[support_ind], y[support_ind], lamb[support_ind]
	w = np.matmul( x_support.T, lamb_support*y_support )
	b = np.mean( y_support - np.matmul( x_support,w ))
	return w,b

np.random.seed(1000)
x, y = load_data(300)
w,b =svm(x,y,max_iter = 800000,plot_training_results = True)
plot_results(x,y, w,b )