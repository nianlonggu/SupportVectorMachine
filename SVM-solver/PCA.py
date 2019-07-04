import numpy as np

def PCA(X, target_dim, return_projection_matrix=False):
	origin_dim = X.shape[0]
	En = np.eye(origin_dim) - 1/ origin_dim * np.ones([ origin_dim, origin_dim ])
	U, sigma, VT = np.linalg.svd( np.matmul(X.T, En ) )
	P = U[:, : target_dim]

	projected_X = np.matmul( X , P )
	projection_matrix = P.T

	if return_projection_matrix:
		return projected_X, projection_matrix
	else:
		return projected_X


