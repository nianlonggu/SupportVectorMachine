from svm import *
from utils import *





def load_data(num_samples = 1000):
	x1 = []
	x2 = []
	for _ in range(num_samples):
		while True:
			r_x = np.random.multivariate_normal( [0,1], [[20,0],[0,1]], 1 )
			if r_x[0,1]>np.sin( r_x[0,0] )+0.5:
				x1.append( r_x )
				break
		while True:
			r_x = np.random.multivariate_normal( [0,-1], [[20,0],[0,1]], 1 )
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



x,y = load_data(500)
x_val, y_val = load_data(100)

estimated_sigma = np.mean( distance_matrix( x,x ) ) * 0.5
print(estimated_sigma)

svm= SVM_Solver( kernel_type = {"name":"GAUSSIAN", "params":[estimated_sigma] } )

epoch =0
while True:
	svm.train(x,y, x_val, y_val, max_iter = 100000)
	epoch +=1
	plot_results( x,y, svm.support_ind, pred_y= svm.predict(x), title = "", img_save_path = generate_folder("results/test_kernel_func/")+"epoch%d.png"%(epoch) , show_img = False )

	if epoch >=20:
		break