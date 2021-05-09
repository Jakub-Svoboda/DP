""" 
Author: Jakub Svoboda
Email:  xsvobo0z@stud.fit.vutbr.cz
School: Brno University of Technology

This code handles the creation and handling of the Database which contains the identity
vectors. The embeddings are held in the 'db' array. This dimension of each embedding is
set to 128 for each, see the thesis text for further information. Each embedding can have 
an optional 'name' parameter attached to it as well.
"""

import numpy as np
from numpy.lib.function_base import angle
import tensorflow as tf

class Database():
	""" This class encapsulates the functionality of the persons' identity database.
	    The embeddings, labels and names are stored as a numpy array.
	"""


	def __init__(self):
		""" Constructor initializes the data storage arrays with appropriate dimensions.
		"""
		self.db = np.zeros(shape = (0, 128), dtype = float)	# Holds the embeddings
		self.labels = np.zeros(shape = (0), dtype = int)	# ID (raising numeral) of the person
		self.names = np.empty(shape = (0), dtype=np.str)	# Optional name of the ID
		
	

	def __str__(self):
		""" Prints the information (shape) of the databae.

		Returns:
			str: Database shape formated for print.
		"""
		if hasattr(self, 'db'):
			return str(self.db.shape)
		else:
			return 'Uninitilized Database'

	

	def findFace(self, embedding):
		""" Searches the database lineary and finds the closest face in DB to given 'embedding'.

		Args:
			embedding (numpy array): The 128 float embedding to be searched for in the DB.

		Returns:
			int: the ID of the closest person in the database.
			str: name of the closest person in the database.
			minAngle: the difference (angle) of the passed embedding and the closest match.
		"""
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
		

	
	def addId(self, embedding, label=None, name=None):
		"""  Appends 'embedding' to the database. 

		Args:
			embedding (128 float numpy array): the embedding which is to be stored.
			label (int, optional): custom ID can be assigned. If none is passed, a new highest ID is assigned. Defaults to None.
			name (str, optional): custon name can be assigned. If none is passed, a space is stored as name. Defaults to None.
		"""
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

	

	def removeId(self, row):	
		""" Removes a row from the datase.
			The person should be specified by its row in the table.
			If the database is empty, no action is taken.

		Args:
			row (int): the index of the row to be deleted.
		"""	
		if self.db.shape[0] == 0:		# if db is empty
			return
		# Remove the embedding, name and ID:
		self.db = np.delete(self.db, row, axis=0)
		self.names = np.delete(self.names, row)	
		self.labels = np.delete(self.labels, row)
		

	
	def _angle(self, e1, e2):
		""" Computes the angle between the 'e1' and 'e2' vector in 128D space.

		Args:
			e1 (128 float numpy array): First embedding.
			e2 (128 float numpy array): Second embedding.

		Returns:
			tf.float: angle between the two normalized vectors.
		"""
		up = tf.multiply(e1, e2)
		up = tf.reduce_sum(up, axis=1)
		# down = tf.tensordot(tf.norm(e1, axis=0), tf.norm(e2, axis=0),2)
		# the embeddings are already normalized, just divide by the sum of their squared values (always 2)
		res = up / 2. 
		# modify the range to <2, 0>
		return tf.math.abs(up - 1.)
