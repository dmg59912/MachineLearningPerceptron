import numpy as np 
from helper import *
from random import seed 
import random

'''
Homework1 for CECS 456: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#
def show_images(data):

			
	'''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
	      images and each image has size 16 by 16. 

	Returns:
		Do not return any arguments, just save the images you plot for your report.
	'''
	#print(data[0])
	#create first matrix from data 16x16
	image1 = []
	for i in range(16):
		row = []
		for j in range(16):
			row.append(0)
		image1.append(row)
	#print(image1)

	for i in range(16):

		for j in range(16):
			image1[i][j] = data[0][i][j]
			#print(data[0][i][j])

	#create second matrix from data 16x16
	image2 = []
	for i in range(16):
		row = []
		for j in range(16):
			row.append(0)
		image2.append(row)
	#print(image1)

	for i in range(16):

		for j in range(16):
			image2[i][j] = data[1][i][j]

	#plot our two images
	f = plt.figure()
	plt.title("Show_images")
	f.add_subplot(1,2, 1)
	plt.imshow(image1)
	f.add_subplot(1,2, 2)
	plt.imshow(image2)
	plt.show(block=True)
	plt.savefig('part one: 2 images', bbox_inches='tight')





def show_features(data, label):
	'''
	This function is used for plot a 2-D scatter plot of the features and save it. 

	Args:
	data: train features with shape (1561, 2). The shape represents total 1561 samples and 
	      each sample has 2 features.
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''

	# create x and y coordiates for red and blue labels 
	label_xr = []
	label_xb = []
	label_yr = []
	label_yb = []

	#check to see if label shape is 1 or -1, then added to x,y coordinates to appropriate label 
	for i in range(1561):
		if label[i] == 1:
			label_xr.append(data[i][0])
			label_yr.append(data[i][1])
		elif label[i] == -1:
			label_xb.append(data[i][0])
			label_yb.append(data[i][1])

	# create our scatter plot with appropriate labels
	plt.scatter(label_xr,label_yr, marker = "*",color = 'r', label = " Label 1")
	plt.scatter(label_xb,label_yb, marker = "+", color = 'b', label = "Label 5")
	plt.title("Show Features!!!")
	plt.ylabel("Average Intensity");
	plt.xlabel("Symmetry");
	plt.legend()
	plt.show()
	plt.savefig('2D scatter plot', bbox_inches='tight')



def perceptron(data, label, max_iter, learning_rate):
	'''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and 5 for digit number -1.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
	seed(1)

	w = np.zeros((1,3))
	count = 0
	#our peceptron function 
	for i in range(1561):
		value = i
		for j in range(max_iter):
			hypothesis = sign(np.dot(data[i],np.transpose(w)))
			if(hypothesis != label[i]):
				w = w + data[value] * label[i] * learning_rate
				hypothesis = sign(np.dot(data[i],np.transpose(w)))
				if (label[i] != hypothesis):
					value = np.random.randint(0,1560)
				if(label[i] == hypothesis):
					break
	return w
		
def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	''' 
	#w transpose x = 0, need to reperesent x and y

	#creating sets of points, one for label 1 and the other for label 5
	x_r = []
	y_r = []

	x_b = []
	y_b = []

	#adding values to our x and y points to be able to plot our scatter points
	for i in range(424):
		if label[i] == 1:
			x_r.append(data[i][0])
			y_r.append(data[i][1])
		elif label[i] == -1:
			x_b.append(data[i][0])
			y_b.append(data[i][1])
	

	x = np.linspace(-.6,0,10)

	m = w[0][1]
	y = m*x + w[0][0]

	#graphing our funtion with a separator 
	plt.scatter(x_r,y_r, marker = "*",color = 'r', label = " Label 1")
	plt.scatter(x_b,y_b, marker = "+", color = 'b', label = "Label 5")
	plt.plot(x,y,'k') #graphs our linear regresion line
	plt.title("Show Result!!!")
	plt.ylabel("Average Intensity");
	plt.xlabel("Symmetry");
	plt.legend()
	plt.show()
	plt.savefig('Show results', bbox_inches='tight')




#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


