"""
Description: A SVM solver
Input:  training dataset (x,y), together with other hype-parameters
Return: a trained SVM model (solver) which is able to perform classification for a give x
"""
import numpy as np

""" kernel part for SVM """
def kernel_func(x1,x2, kernel_type=None):
	if kernel_type is None:
		return np.dot( x1,x2)
	elif kernel_type["name"]=="GAUSSIAN":
		sigma = kernel_type["params"][0]
		return np.exp(- np.dot( x1-x2, x1-x2 )/(2*sigma**2)  )



def get_kernel_matrix( x1, x2, kernel_type=None ):
	num_samples_x1 = x1.shape[0]
	num_samples_x2 = x2.shape[0]
	kernel_matrix = np.zeros([num_samples_x1, num_samples_x2])
	for nrow in range(num_samples_x1 ):
		for ncol in range(num_samples_x2  ):
			kernel_matrix[nrow][ncol] = kernel_func(x1[nrow] , x2[ncol], kernel_type = kernel_type)
	return kernel_matrix




class SVM_Solver:
	def __init__(self, kernel_type=None , C=10):

		self.support_ind = None
		self.support_x = None
		self.support_y = None
		self.support_lamb = None
		self.kernel_type= kernel_type
		self.C = C
		self.count = 0
		self.objective_func = -np.Inf
		self.lamb = None
		self.param_b = None

	## This is a SVM trained predictor
	def predict(self,x, decision_mode = "hard"):
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
		K = get_kernel_matrix(self.support_x, x, kernel_type = self.kernel_type )
		pred_y = []
		for ind in range(x.shape[0]):
			z= np.dot( self.support_lamb* self.support_y, K[:,ind] ) +  self.param_b  
			pred_y.append(decision_func(z))	
		return np.array(pred_y)
		
		"""Training the SVM model, which uses x, y and validation set x_val, y_val
        max_iter is the maximum iteration to train;
        epsilon is use to determine when the training is terminated -- the change of objective
        function is less than epsilon
        """
	def train( self, x, y, x_val, y_val, max_iter= 1E6, epsilon= 1E-4 ):
	
		num_samples = x.shape[0]
		"""Solve the dual problem using SMO"""
		## Initialization
		K=get_kernel_matrix(x,x, kernel_type = self.kernel_type )	
		C = self.C
		if self.lamb is None:
			self.lamb = np.zeros(num_samples)
		if self.param_b is None:
			self.param_b = np.random.normal()
		## Start looping:
		## looping parameters:

		local_count =0
		##Here is the part of the SMO algorithm
		while True:
			## randomly select a pair (a,b) to optimize
			[a,b] = np.random.choice( num_samples, 2, replace= False )
			if K[a,a] + K[b,b] - 2*K[a,b] ==0:
				continue	

			lamb_a_old = self.lamb[a]
			lamb_b_old = self.lamb[b]	

			Ea =  np.dot(self.lamb * y, K[:,a]) + self.param_b - y[a]
			Eb =  np.dot(self.lamb * y, K[:,b]) + self.param_b - y[b]	

			lamb_a_new_unclip = lamb_a_old  + y[a] *(Eb-Ea)/( K[a,a] + K[b,b] - 2*K[a,b] )
			xi = - lamb_a_old  * y[a] - lamb_b_old * y[b]	

			if y[a] != y[b]:
				L = max( xi * y[b], 0 )
				H = min( C+xi*y[b], C )
			else:
				L = max( 0, -C-xi*y[b])
				H = min( C, -xi*y[b] )	

			if lamb_a_new_unclip < L:
				lamb_a_new = L
			elif lamb_a_new_unclip > H:
				lamb_a_new = H
			else:
				lamb_a_new = lamb_a_new_unclip	

			lamb_b_new = lamb_b_old + ( lamb_a_old - lamb_a_new )*y[a] * y[b]
			if lamb_a_new >0 and lamb_a_new <C:
				self.param_b =  self.param_b - Ea + ( lamb_a_old- lamb_a_new)*y[a]*K[a,a] + (lamb_b_old - lamb_b_new)*y[b] * K[b,a]
			elif lamb_b_new >0 and lamb_b_new <C:
				self.param_b = self.param_b - Eb + ( lamb_a_old- lamb_a_new)*y[a]*K[a,b] + (lamb_b_old - lamb_b_new)*y[b] * K[b,b]	

			self.lamb[a] = lamb_a_new
			self.lamb[b] = lamb_b_new	

			self.count +=1
			local_count +=1

			"""Every 5000 iterations record the current progree of the training,
            and determine whether to stop the training.
            """
			if local_count >= max_iter or self.count % 5000 ==0:
				## get the support set
				self.support_ind =  self.lamb > 0
				self.support_x = x[self.support_ind]
				self.support_y = y[self.support_ind]
				self.support_lamb = self.lamb[self.support_ind]	
	
				## Evaluate the performance (accuracy) on training set and validation set
				pred_y=self.predict(x)
				train_acc =  np.sum( pred_y == y)/ y.shape[0]
				pred_y=self.predict(x_val)
				val_acc =  np.sum( pred_y == y_val  )/ y_val.shape[0]

				support_K = K[ self.support_ind,: ][:, self.support_ind]
				new_objective_func = np.sum( self.support_lamb ) - 0.5 * np.dot( np.matmul( ( self.support_lamb *self.support_y ).T, support_K ).T , self.support_lamb* self.support_y  ) 

				## support ratio represents the percentage of the points which are support vectors
				support_ratio = np.sum( self.support_ind )/ self.support_ind.shape[0] 

				print("Iteration: %d, \tTrain accuracy: %.2f%%, \tVal accuracy: %.2f%%, \tDelta Objective Function: %f, \tSupport Ratio: %.2f%%"%(self.count, train_acc*100, val_acc*100, new_objective_func - self.objective_func, support_ratio *100 ))
				
				## If the change of dual objective function is less than epsilon, then stop training
				if abs( new_objective_func - self.objective_func ) <= epsilon:
					break
				else:
					self.objective_func = new_objective_func
				
				if local_count >= max_iter:
					break










