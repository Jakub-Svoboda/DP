import numpy as np
import pickle
import os
from numpy.lib.function_base import angle
import tensorflow as tf

class Database():

	def __init__(self):
		self.db = np.zeros(shape = (0, 128), dtype = float)
		self.labels = np.zeros(shape = (0), dtype = int)
		self.names = np.empty(shape = (0), dtype=np.str)
		

	def __str__(self):
		if hasattr(self, 'db'):
			return str(self.db.shape)
		else:
			return 'Uninitilized Database'

	def findFace(self, embedding):
		minAngle = 1000
		minId = None
		for idx, face in enumerate(self.db):
			face = np.expand_dims(face, 0)
			dist = self._angle(embedding, face)
			print(self.names[idx], dist)
			if dist < minAngle:
				minAngle = dist
				minId = idx
		return minId, self.names[minId], minAngle
		


	def addId(self, embedding, label=None, name=None):
		self.db = np.concatenate((self.db, embedding))
		if label is None:
			self.labels = np.append(self.labels, self.labels.shape[0])
		else:
			self.labels = np.append(self.labels, label)
		if name is None:
			self.names = np.append(self.names, ' ')
		else:
			self.names = np.append(self.names, name)

	def _angle(self, e1, e2):
		#print(e1.shape, e2.shape)
		up = tf.multiply(e1, e2)
		up = tf.reduce_sum(up, axis=1)
		#down = tf.tensordot(tf.norm(e1, axis=0), tf.norm(e2, axis=0),2)
		# the embeddings are already normalized, just divide by the sum of their squared values (always 2)
		res = up / 2. 
		# modify the range to <2, 0>
		#return 1. - res 
		return tf.math.abs(up - 1.)
