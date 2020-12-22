import numpy as np
import cv2
import sys
import os
from imutils import face_utils
import argparse
import imutils
import dlib
import pickle
from sklearn.preprocessing import scale
import random


class Dataset():
	

	def __init__(self):
		self.faceDataset = np.zeros(shape = (0, 68, 2), dtype = float)
		self.identities = np.zeros(shape = (0), dtype = int)

		self.facePairs = np.zeros(shape = (0, 2, 68, 2), dtype = float)
		self.labels = np.array([])



	def __str__(self):
		toPrint = "Dataset contains: " + str(self.faceDataset.shape[0]) + " faces and " + str(self.identities.shape[0]) + " identities."
		toPrint += "\nFor training there are " + str(self.facePairs.shape[0]) + " image pairs with " +  str(self.labels.shape[0]) + " labels."
		return toPrint

	def loadDataset(self, path="dataset.pickle"):
		(self.faceDataset, self.identities) = pickle.load(open(path, 'rb'))
		

	def saveDataset(self, path="dataset.pickle"):
		pickle.dump( (self.faceDataset, self.identities), open(path, 'wb'))

	def addFace(self, shape, label):
		#Calculate the width and the height of the facial points
		maxX, maxY = np.max(shape, axis=0)
		minX, minY = np.min(shape, axis=0)
		width = maxX - minX
		height = maxY - minY
		maxDimensionSize = max(width, height)
		#Substract the minimal value from each column to move the points to the edge
		col1 = shape[:,0]
		col2 = shape[:,1]
		col1 -= minX
		col2 -= minY
		shape = np.column_stack((col1, col2))
		#convert to floats and normalize between 0 and 1
		shape = shape.astype(float)	
		shape = np.divide(shape, maxDimensionSize) 
		shape = np.expand_dims(shape, axis=0)
		#Add new face to the dataset
		self.faceDataset = np.concatenate((self.faceDataset, shape))
		self.identities = np.concatenate((self.identities, np.array([label])))
		
	def createPairs(self):

		for idx, image in enumerate(self.faceDataset):	#for each image in dataset
			#get image with different identity
			filteredFaces = self.faceDataset[self.identities != self.identities[idx]]
			filteredIDs = self.identities[self.identities != self.identities[idx]]
			#get random index
			n = random.randint(0,filteredFaces.shape[0] -1)
			pairedDifferent = filteredFaces[n]
			pairedId = filteredIDs[n]
			#print("selected randnum",n, "orig identity:" , self.identities[idx],"Paired image has identity", pairedId)
			pair = np.stack((image, pairedDifferent))	#concatenate the two images side by side
			pair = np.expand_dims(pair, axis=0)
			self.facePairs = np.concatenate((self.facePairs, pair), axis = 0)
			#print(self.facePairs.shape)
			self.labels = np.concatenate((self.labels, np.array([-1])))		#Different identity label == -1


			#get an image with the same ID
			filteredFaces = self.faceDataset[self.identities == self.identities[idx]]
			filteredIDs = self.identities[self.identities == self.identities[idx]]
			n = random.randint(0,filteredFaces.shape[0] -1)
			pairedSame = filteredFaces[n]
			pairedId = filteredIDs[n]
			#print("selected randnum",n, "orig identity:" , self.identities[idx],"Paired dif image has identity", pairedId)
			pair = np.stack((image, pairedSame))	#concatenate the two images side by side
			pair = np.expand_dims(pair, axis=0)
			self.facePairs = np.concatenate((self.facePairs, pair), axis = 0)
			#print(self.facePairs.shape)
			self.labels = np.concatenate((self.labels, np.array([1])))		#Same identity label == 1

	def findFace(self, face):
		pass



















