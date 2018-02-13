import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image
import numpy as np
import glob
import random
from classificator import *



class work1:

	name = ""
	list_chosen_class	= [] 			 #List of the chosen class
	list_images_path	= []			 #List of the images path
	list_images_matrix	= np.asarray([]) #List of the images matrix	
	list_lable_class	= []
	X = []
	X_t = []
	pca = []
	static_path = '/Users/Host/Desktop/Sapienza/MARR/src/ML/coil-100'
	obj 		=  "/obj"
	obj_end 	= "__*"


	#Constructor 
	def __init__(self,Name):
		self.name = ""
		self.list_chosen_class	= [] 			 
		self.list_images_path	= []			 
		self.list_images_matrix	= np.asarray([]) 
		self.list_lable_class	= []
		self.X = []
		self.X_t = []
		self.list_chosen_class = [1,2,3,4]
		self.name = Name
		print("########  "+self.name+"  ########")
		#hw1.random_class()
		self.load_all_image_path()
		self.load_all_class_label() 
		self.load_images_matrix()



	#Choose 4 randome class 
	def random_class(self):
		self.list_chosen_class = [self.rand_num(),self.rand_num(),self.rand_num(),self.rand_num()]

	#Add single image path to < list_images_path >
	def load_single_image_path(self,num_obj):
		temp_path = self.obj + str(self.list_chosen_class[num_obj]) + self.obj_end
		print("Path loaded - Class "+str(self.list_chosen_class[num_obj])+" .")
		self.list_images_path.extend(glob.glob(self.static_path+temp_path))


	#Call  n = len(self.list_chosen_class) numero delle classi scelte add_single_image_path
	def load_all_image_path(self):
		for obj_num in range(0,len(self.list_chosen_class)):
			self.load_single_image_path(obj_num)



	#Load images into < list_images_matrix >
	def load_images_matrix(self):
		self.list_images_matrix = np.asarray([np.asarray(Image.open(self.list_images_path[i]).convert('L'), 'f') for i in range(len(self.list_images_path))])
		print("Image matrix loaded - list_images_matrix len = "+str(len(self.list_images_matrix))+" .")
		


	#Append 72 times the vlaue to < list_lable_class >
	def load_single_class_label(self,value):
			for i in range(0,72): 
				self.list_lable_class.append(value)



	#Call for the number of class chosen load_single_class_label
	def load_all_class_label(self):
		for obj in range(1,len(self.list_chosen_class)+1):
			self.load_single_class_label(obj)


	#Give random value betwen 1 - 99
	def rand_num(self):
		return random.randrange(1,99)

	#Return X_t
	def get_X(self):
		return self.X_t

	#Return Y_t
	def get_Y(self):
		return self.list_lable_class

	#ravel create from a (288,128,128) to matrix a (4718592,) vector
	#model X and create from (4718592,) vector a (288, 16384) 
	#unity-variance #zero-mean
	def X_value(self):
		self.X = self.list_images_matrix.ravel()  	
		self.X = np.array(self.X).reshape(288, -1)		
		self.X = preprocessing.scale(self.X)				

	# PCA(2) compute the first and second PCA vectors, fit.transform(X)
	def do_pca(self,value):
		self.pca = PCA(n_components=value)
		self.X_t = self.pca.fit_transform(self.X)				

	#Plot with scatter the X_t[:, value1],X_t[:, value2]
	def plot(self,value1,value2):
		#style= plt.style.available
		#plt.style.use(style[1])
		plt.title("PCA - "+self.name)
		plt.plot(self.X_t[0:71,value1],self.X_t[0:71,value2], 'o', markersize=4, color='blue', alpha=0.95, label='class1')
		plt.plot(self.X_t[71:143,value1],self.X_t[71:143,value2], 'o', markersize=4, color='green', alpha=0.95, label='class2')
		plt.plot(self.X_t[143:215,value1],self.X_t[143:215,value2], 'o', markersize=4, color='orange', alpha=0.95, label='class3')
		plt.plot(self.X_t[215:287,value1],self.X_t[215:287,value2], 'o', markersize=4, color='red', alpha=0.95, label='class4')
		plt.xlabel('Principal Componet '+str(value1))
		plt.ylabel('Principal Componet '+str(value2))
		plt.legend()
		plt.tight_layout()
		#plt.scatter(self.X_t[:, value1], self.X_t[:, value2] ,c=self.list_lable_class)		#plot
		plt.show()
		plt.close()



	def get_only_coloum_X_t(self,colum1,colum2):
		self.X_t = self.X_t[:,[colum1,colum2]]

	def plot_variance(self):
		cum_var_exp  = [0]
		cum_var_exp = np.append(cum_var_exp,self.pca.explained_variance_.cumsum())
		
		## Subplot2Grid Layout
		ax1 = plt.subplot2grid((12,1),(0,0), rowspan=3, colspan=1)
		ax2 = plt.subplot2grid((12,1),(4,0), rowspan=3, colspan=1)
		ax3 = plt.subplot2grid((12,1),(8,0), rowspan=3, colspan=1)

		## Cumulative Variance 
		last = len(cum_var_exp)-1
		cum_var_exp = self.get_componet(cum_var_exp,last-1,last)
		ax1.step(range(0,len(cum_var_exp)), cum_var_exp/100,label='cumulative explained variance')
		ax1.set_title("Cumulative - Variance ")
		#ax1.set_ylim(0, 100)
		ax1.set_xlabel('n Componets')
		ax1.set_ylabel('% of Variance explained')


		# Individual variance
		last = len(self.pca.explained_variance_)-1
		ind_var_exp = self.get_componet(self.pca.explained_variance_,last-1,last)
		
		ax2.bar(range(len(ind_var_exp)), ind_var_exp/100, alpha=0.5, align='center',
			label='individual explained variance')
		ax2.set_title("Individual - Variance")
		ax2.set_xlabel('n Componets')
		ax2.set_xlim(0,len(ind_var_exp))
		ax2.set_ylabel('% of Variance explained')

	
		# Ratio of variance
		last = len(self.pca.explained_variance_ratio_)-1
		rat_var_exp = self.get_componet(self.pca.explained_variance_ratio_,last-1,last)
		ax3.set_title("Ratio of Variance")
		ax3.set_xlabel('n Componets')
		ax3.set_ylabel('% of Variance explained')
		ax3.plot(rat_var_exp,label='ratio explained variance')
		
		plt.show()
		plt.close()

	#def get_only_componets_variance(self):

	def get_componet(self,array,val1,val2):
		temp = np.asarray([])
		temp = np.append(temp,array[val1])
		temp = np.append(temp,array[val2])
		return temp




	





