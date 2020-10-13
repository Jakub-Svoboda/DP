"""
	Database manager for facial landmarks
	Author: Jakub Svoboda
	Date 6.10.2020
	Email: xsvobo0z@stud.fit.vutbr.cz
"""

import numpy as np
import cv2
import sys
import os
import argparse
import imutils
import dlib
import multiprocessing
from multiprocessing import Process, Queue, Pool
import cProfile
import warnings
import csv

import datasetManager


def loadFaces(queue, arguments, folderpath, trainList, splitStart, splitEnd, labelList, limiter = 100000):
	dataset = datasetManager.Dataset()
	
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(arguments["shape_predictor"])

	counter = splitStart
	for picname in trainList[splitStart:splitEnd]:
		if counter == splitEnd:
			break
		#if we want to load only a limited dataset:
		if  labelList[counter] > limiter:	#if the identity of the person is above a certain number, skip it
			counter+=1
			continue
		else:
			print(multiprocessing.current_process(), "picked image", picname, "with indentity", labelList[counter])

		# Load an color image in grayscale
		path = os.path.join(folderpath, trainList[counter])

		#print(path)
		img = cv2.imread(path)

		image = imutils.resize(img, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale image
		rects = detector(gray, 1)


		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = imutils.face_utils.shape_to_np(shape)
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			#(x, y, w, h) = imutils.face_utils.rect_to_bb(rect)

			#add the face to the dataset, along with its label
			dataset.addFace(shape, labelList[counter])
		counter+=1
	queue.put(dataset)	
	return dataset	


def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=False,
		help="path to input image")
	arguments = vars(ap.parse_args())

	#create dataset
	dataset = datasetManager.Dataset()

	#load images
	path = os.path.join("celebA","Img", "img_align_celeba")

	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
		break

	trainList = filenames[:162770]
	valList = filenames[162770:182638]
	testList = filenames[182638:]
	print(str(len(trainList)) + " train images")
	print(str(len(valList)) + " validation images")
	print(str(len(testList)) + " test images")

	#load labels
	labelPath = os.path.join("celebA","Anno", "identity_CelebA.txt")
	labels = []
	with open(labelPath) as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ')
		for row in spamreader:
			labels.append(int(row[1]))

	#start multiprocessing for image processing
	procs = []
	rets = []
	numOfProcs = multiprocessing.cpu_count()
	imagesPerProcess = 162770 // numOfProcs		#get the number that each processor has to parse
	#imagesPerProcess = 20//numOfProcs
	q = Queue()
	for num in range(0,numOfProcs):
		print(num, imagesPerProcess*num, imagesPerProcess*(num+1))
		p = Process(target=loadFaces, args=(q, arguments, path, trainList, imagesPerProcess*num, imagesPerProcess*(num+1), labels, 100))
		p.start()
		procs.append(p)

	for p in procs:
		ret = q.get() # will block
		rets.append(ret)
	for p in procs:
		p.join()

	#merge the partial datasets
	dataset.faceDataset = rets[0].faceDataset
	dataset.labels = rets[0].labels
	for num in range(1,numOfProcs): 
		dataset.faceDataset = np.concatenate((dataset.faceDataset, rets[num].faceDataset))
		dataset.labels = np.concatenate((dataset.labels, rets[num].labels))
	
	dataset.saveDataset()
	print(dataset)

if __name__ == "__main__":
	main()