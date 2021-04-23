##############################################
# Author: Jakub Svoboda
# Email:  xsvobo0z@stud.fit.vutbr.cz
# School: Brno University of Technology
##############################################
# This code handles the creation and handling of the Database which contains the identity
# vectors. The embeddings are held in the 'db' array. This dimension of each embedding is
# set to 128 for each, see the thesis text for further information. Each embedding can have 
# an optional 'name' parameter attached to it as well.
##############################################

import numpy as np
from numpy.lib.function_base import angle
import tensorflow as tf

class Database():

	# Constructor
	def __init__(self):
		self.db = np.zeros(shape = (0, 128), dtype = float)	# Holds the embeddings
		self.labels = np.zeros(shape = (0), dtype = int)	# ID (raising numeral) of the person
		self.names = np.empty(shape = (0), dtype=np.str)	# Optional name of the ID
		
	
	# Prints the information (shape) of the databae
	def __str__(self):
		if hasattr(self, 'db'):
			return str(self.db.shape)
		else:
			return 'Uninitilized Database'

	# Searches the database lineary and finds the closest face in DB to given 'embedding'
	def findFace(self, embedding):
		minAngle = 1000	# init best angle
		minId = None	# init closes embedding ID
		for idx, face in enumerate(self.db):		# go through each face in DB
			face = np.expand_dims(face, 0)			
			dist = self._angle(embedding, face)		# get distance between embedding and 
			print(self.names[idx], dist)
			if dist < minAngle:
				minAngle = dist
				minId = idx
		return minId, self.names[minId], minAngle
		

	# Appends 'embedding' to the database.
	def addId(self, embedding, label=None, name=None):
		self.db = np.concatenate((self.db, embedding))	# append embedding
		if label is None:
			if self.labels.shape[0] > 0:		# if db is not empty
				self.labels = np.append(self.labels, np.amax(self.labels) +1)	# set to new highest number
			else:
				self.labels = np.append(self.labels, 0)	# if its the first Id added to an empty DB, add it as zero
		else:
			self.labels = np.append(self.labels, label)
		if name is None:								# optionally set name
			self.names = np.append(self.names, ' ')
		else:
			self.names = np.append(self.names, name)


	# Computes the angle between the 'e1' and 'e2' vector in 128D space.
	def _angle(self, e1, e2):
		up = tf.multiply(e1, e2)
		up = tf.reduce_sum(up, axis=1)
		# down = tf.tensordot(tf.norm(e1, axis=0), tf.norm(e2, axis=0),2)
		# the embeddings are already normalized, just divide by the sum of their squared values (always 2)
		res = up / 2. 
		# modify the range to <2, 0>
		return tf.math.abs(up - 1.)
