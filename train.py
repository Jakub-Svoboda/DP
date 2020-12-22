"""
	Traning script for facial landmarks
	Author: Jakub Svoboda
	Date 6.10.2020
	Email: xsvobo0z@stud.fit.vutbr.cz
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

import datasetManager



def main():
	#create dataset
	dataset = datasetManager.Dataset()
	dataset.loadDataset()
	
	#create a list of comparison images so that each network can train on 2 images
	dataset.createPairs()
	print(dataset)
	x_train = dataset.facePairs
	y_train = dataset.labels



	# Add a channels dimension
	#x_train = x_train[..., tf.newaxis].astype("float32")
	#x_test = x_test[..., tf.newaxis].astype("float32")
	
	train_ds = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).shuffle(10000).batch(32)
	
	#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	#test_loss = tf.keras.metrics.Mean(name='test_loss')
	#test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	input_shape = (136)
	left_input = keras.layers.Input(input_shape)
	right_input = keras.layers.Input(input_shape)
	# Create an instance of the model
	model = keras.Sequential()
	model.add(Dense(136, activation="sigmoid"))
	model.add(Dense(136, activation="sigmoid"))

	#call the convnet Sequential model on each of the input tensors so params will be shared
	encoded_l = model(left_input)
	encoded_r = model(right_input)
	#layer to merge two encoded inputs with the l1 distance between them
	L1_layer = keras.layers.Lambda(lambda tensors:keras.backend.abs(tensors[0] - tensors[1]))
	#call this layer on list of two input tensors.
	L1_distance = L1_layer([encoded_l, encoded_r])
	prediction = Dense(1,activation='sigmoid')(L1_distance)
	siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

	
	@tf.function
	def train_step(images, labels):
		with tf.GradientTape() as tape:
			# training=True is only needed if there are layers with different
			# behavior during training versus inference (e.g. Dropout).
			predictions = model(images, training=True)
			loss = loss_object(labels, predictions)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

	@tf.function
	def test_step(images, labels):
		# training=False is only needed if there are layers with different
		# behavior during training versus inference (e.g. Dropout).
		predictions = model(images, training=False)
		t_loss = loss_object(labels, predictions)

		test_loss(t_loss)
		test_accuracy(labels, predictions)

	EPOCHS = 5

	for epoch in range(EPOCHS):
		print("\U0001F449\U0001F600\U0001F449")
		# Reset the metrics at the start of the next epoch
		train_loss.reset_states()
		train_accuracy.reset_states()
		#test_loss.reset_states()
		#test_accuracy.reset_states()

		for images, labels in train_ds:
			train_step(images, labels)

		#for test_images, test_labels in test_ds:
		#	test_step(test_images, test_labels)

		print(
			f'Epoch {epoch + 1}, '
			f'Loss: {train_loss.result()}, '
			f'Accuracy: {train_accuracy.result() * 100}, '
			#f'Test Loss: {test_loss.result()}, '
			#f'Test Accuracy: {test_accuracy.result() * 100}'
		)

	
	exit()



if __name__ == "__main__":
	main()