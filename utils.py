import numpy as np

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
