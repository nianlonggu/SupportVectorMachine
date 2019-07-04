import numpy as np
import matplotlib.pyplot as plt
import os

def generate_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

	return path


def load_data(num_samples = 1000):
	x1 = []
	x2 = []
	for _ in range(num_samples):
		while True:
			r_x = np.random.multivariate_normal( [0,1], [[2,0],[0,1]], 1 )
			if r_x[0,1]>np.sin( r_x[0,0] )+0.5:
				x1.append( r_x )
				break
		while True:
			r_x = np.random.multivariate_normal( [0,-1], [[2,0],[0,1]], 1 )
			if r_x[0,1]<np.sin( r_x[0,0] ):
				x2.append( r_x )
				break


	x1 = np.concatenate( x1, axis =0 )
	x2 = np.concatenate( x2, axis =0)
	y1 = np.ones([num_samples]) *-1
	y2 = np.ones([num_samples]) *1
	x = np.concatenate([x1,x2], axis =0)
	y = np.concatenate([y1,y2], axis =0)

	return x, y

def distance_matrix( X,Y, metric = "Euclidean" ):
	def distance( x,y ):
		if metric == "Euclidean":
			return np.linalg.norm(x-y)
	n_row = X.shape[0]
	n_col = Y.shape[0]
	dis_matrix = np.zeros([n_row, n_col] )
	for r in range( n_row ):
		for c in range(n_col ):
			dis_matrix[r][c] = distance( X[r], Y[c])
	return dis_matrix

def within_class_average_distance(x,y):
	x_pos = x[y==1]
	x_neg = x[y==-1]
	dis_matrix_pos = distance_matrix(x_pos,x_pos)
	dis_matrix_neg = distance_matrix(x_neg, x_neg)
	return (np.mean(dis_matrix_pos)+np.mean(dis_matrix_neg))/2

def plot_results( x,y, params, classifier, title = "", img_save_path = None , show_img = True ):

	fig, ax = plt.subplots()
	pred_y = classifier(x)

	x_low_dim = x


	x_support =  x[params["support_ind"]]
	y_support = y[params["support_ind"]]
	pred_y_support = pred_y[params["support_ind"]]
	x_support_low_dim = x_low_dim[params["support_ind"]]


	for ind in range(x.shape[0]):
		if y[ind] == 1:
			mshape = "^"
		else:
			mshape = "o"
		if pred_y[ind] == 1:
			color = "r"
		else:
			color = "b"

		plt.plot(x_low_dim[ind,0], x_low_dim[ind,1], mshape, c= color, markerfacecolor='none', markeredgewidth=0.4, markersize =4)



	for ind in range(x_support.shape[0]):
		if y_support[ind] == 1:
			mshape = "^"
		else:
			mshape = "o"
		if pred_y_support[ind] == 1:
			color = "r"
		else:
			color = "b"

		plt.plot(x_support_low_dim[ind,0], x_support_low_dim[ind,1], mshape, c= color, markersize =4)


	for ind in range(x.shape[0]):
		if y[ind]!= pred_y[ind]:
			plt.plot(x_low_dim[ind,0], x_low_dim[ind,1], "o", c= "g", markersize =9, markerfacecolor='none')


	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim([min(x_low_dim[:,0])-0.5, max(x_low_dim[:,0])+0.5 ])
	plt.ylim([min(x_low_dim[:,1])-0.5, max(x_low_dim[:,1])+0.5 ])

	plt.title(title)
	if img_save_path is not None:
		plt.savefig( img_save_path )
	if show_img:
		plt.show()

	plt.close()


""" kernel part for SVM """
def kernel_func(x1,x2, kernel_type=None):
	if kernel_type is None:
		return np.dot( x1,x2)
	elif kernel_type["name"]=="GAUSSIAN":
		sigma = kernel_type["params"][0]
		return np.exp(- np.dot( x1-x2, x1-x2 )/(2*sigma**2)  )



def get_kernel_matrix( x, kernel_type=None ):
	num_samples = x.shape[0]
	kernel_matrix = np.zeros([num_samples, num_samples])
	for nrow in range(num_samples):
		for ncol in range(num_samples):
			kernel_matrix[nrow][ncol] = kernel_func(x[nrow] , x[ncol], kernel_type = kernel_type)
	return kernel_matrix


def smo(kernel_matrix, y, lamb, b  ,max_iter=10000,  C = np.Infinity  ):
	num_samples = kernel_matrix.shape[0]

	current_iter = 0
	while True:
		[inda, indb] = np.random.choice(num_samples, 2, replace = False)
		Ea = np.dot( lamb * y, kernel_matrix[:,inda]) + b - y[inda]
		Eb = np.dot( lamb * y, kernel_matrix[:, indb]) +b - y[indb] 
		Eta = kernel_matrix[inda][inda] + kernel_matrix[indb][indb] - 2* kernel_matrix[inda][indb]

		if Eta == 0:
			continue

		lamba_new_unclipped = lamb[inda] + y[inda]*(Eb-Ea)/Eta

		## Now we deal with the Bosk constraints
		if y[inda] == y[indb]:
			L = max( 0, lamb[inda]+lamb[indb] - C )
			H = min( C, lamb[inda]+lamb[indb] )
		else:
			L = max( 0, lamb[inda]-lamb[indb] )
			H = min( C, lamb[inda]-lamb[indb] + C )

		## get the clipped new lamba
		if lamba_new_unclipped < L:
			lamba_new = L
		elif lamba_new_unclipped > H:
			lamba_new = H
		else:
			lamba_new = lamba_new_unclipped

		lambb_new = lamb[indb] + y[inda]*y[indb]*( lamb[inda] - lamba_new )
		
		lamba_old = lamb[inda]
		lambb_old = lamb[indb]

		lamb[inda] = lamba_new
		lamb[indb] = lambb_new

		## update b
		# ba = y[inda] - np.dot( lamb*y, kernel_matrix[:, inda] ) 
		# bb = y[indb] - np.dot( lamb*y, kernel_matrix[:, indb] )
		ba = b - Ea + ( lamba_old- lamba_new )*y[inda]*kernel_matrix[inda][inda] + (lambb_old - lambb_new)*y[indb]*kernel_matrix[indb][inda]
		bb = b - Eb + ( lamba_old- lamba_new )*y[inda]*kernel_matrix[inda][indb] + (lambb_old - lambb_new)*y[indb]*kernel_matrix[indb][indb]

		if lamba_new >0 and lamba_new < C:
			b = ba
		elif lambb_new >0 and lambb_new < C:
			b = bb

		current_iter +=1
		if current_iter >= max_iter:
			break
	return lamb, b


def svm_classifier( x_support, y_support, lamb_support, b,  input_x, kernel_type= None , decision_mode= "hard" ):
	input_x = np.array(input_x)
	assert len(input_x.shape)==2 and len(x_support.shape)==2

	def decision_func(z):
		if decision_mode == "soft":
			if z<-1:
				return -1
			elif z>1:
				return 1
			else:
				return z
		elif decision_mode == "hard":
			if z<0:
				return -1
			else:
				return 1

	pred_y=[]
	for ind in range( input_x.shape[0] ):
		z=0
		for j in range(x_support.shape[0]):
			z+= lamb_support[j] * y_support[j] * kernel_func( x_support[j], input_x[ind], kernel_type = kernel_type )
		z += b
		pred_y.append(decision_func(z))

	return np.array(pred_y)


def train_svm( x,y, C=np.Infinity, kernel_type = None, max_epoch =100 , smo_iter_per_epoch = 10000 ,epsilon = 1e-7 , plot_training_results= False ):

	kernel_matrix = get_kernel_matrix(x, kernel_type = kernel_type )
	## initialization
	lamb = np.zeros( x.shape[0] )
	b = np.random.normal()

	for epoch in range( max_epoch ):
		lamb, b = smo( kernel_matrix, y, lamb, b, max_iter = smo_iter_per_epoch, C=C )

		print("epoch %d"%(epoch))

		if plot_training_results:
			support_ind= np.argwhere(lamb>1e-7)[:,0]
			x_support, y_support, lamb_support = x[support_ind], y[support_ind], lamb[support_ind]
			params={
			"support_ind": support_ind,
			"x_support": x_support,
			"y_support": y_support,
			"lamb_support": lamb_support,
			"b": b}
			def my_classifier( input_x, decision_mode="hard"):
				return svm_classifier( x_support, y_support, lamb_support, b,  input_x, kernel_type= kernel_type , decision_mode= decision_mode )
			plot_results(x,y, params, my_classifier,title= "epoch %d"%(epoch), img_save_path=generate_folder("results/smo-svm/")+"results-epoch%d.jpg"%(epoch), show_img=False  )

	## only return support vectors parameters:
	support_ind= np.argwhere(lamb>1e-7)[:,0]
	x_support, y_support, lamb_support = x[support_ind], y[support_ind], lamb[support_ind]

	def my_classifier( input_x, decision_mode="hard"):
		return svm_classifier( x_support, y_support, lamb_support, b,  input_x, kernel_type= kernel_type , decision_mode= decision_mode )

	params={
			"support_ind": support_ind,
			"x_support": x_support,
			"y_support": y_support,
			"lamb_support": lamb_support,
			"b": b}

	return params, my_classifier



x, y = load_data(300)




params , my_classifier =  train_svm(x,y,C=10,  plot_training_results=True, kernel_type = {"name":"GAUSSIAN", "params":[ within_class_average_distance(x,y) ]})
print(my_classifier( [[1,2]] ))
