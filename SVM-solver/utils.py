import numpy as np
import matplotlib.pyplot as plt
import os

def distance_matrix( x,y, metric = "Euclidean" ):
	def distance( a,b ):
		if metric == "Euclidean":
			return np.linalg.norm(a-b)
	n_row = x.shape[0]
	n_col = y.shape[0]
	dis_matrix = np.zeros([n_row, n_col] )
	for r in range( n_row ):
		for c in range(n_col ):
			dis_matrix[r][c] = distance( x[r], y[c])
	return dis_matrix


def generate_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

	return path

def plot_results( x,y, support_ind, pred_y, title = "", img_save_path = None , show_img = True ):

	fig, ax = plt.subplots()

	# x_low_dim, P = PCA(x, 2, return_projection_matrix = True)
	x_low_dim = x

	x_support =  x[support_ind]
	y_support = y[support_ind]
	pred_y_support = pred_y[support_ind]
	x_support_low_dim = x_low_dim[support_ind]


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