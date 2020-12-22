import tensorflow as tf


def getModel():
	model = tf.keras.Sequential([
	# Block 1
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(28,28,1)),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
	#tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),

	# Block 2
	#tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
	#tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
	#tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),

	# Block 3
	#tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
	#tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
	#tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
	#tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),

	# Block 4
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
	#tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),

	# Block 5
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
	#tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
	#tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),


	# Classification block
	tf.keras.layers.Flatten(name='flatten'),
	tf.keras.layers.Dense(256, activation='relu', name='fc1'),
	#tf.keras.layers.Dense(4096, activation='relu', name='fc2')
	tf.keras.layers.Dense(128, activation=None), # No activation on final dense layer
	tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
	])

	return model
	