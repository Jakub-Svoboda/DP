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


class Dataset():
	

	def __init__(self):
		self.faceDataset = np.zeros(shape = (0, 68, 2), dtype = float)
		self.labels = np.zeros(shape = (0), dtype = int)

	def __str__(self):
		toPrint = "Dataset contains: " + str(self.faceDataset.shape[0]) + " faces\n" + "and " + str(self.labels.shape[0]) + " labels."
		return toPrint

	def loadDataset(self, path="dataset.pickle"):
		self = pickle.load(open(path, 'rb'))

	def saveDataset(self, path="dataset.pickle"):
		pickle.dump(self, open(path, 'wb'))

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
		self.labels = np.concatenate((self.labels, np.array([label])))
		
	
	def findFace(self, face):
		pass



















