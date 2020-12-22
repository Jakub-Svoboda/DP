import tensorflow as tf


def getModel():
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
	tf.keras.layers.MaxPooling2D(pool_size=2),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
	tf.keras.layers.MaxPooling2D(pool_size=2),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(256, activation="relu"), 
	tf.keras.layers.Dense(128, activation=None), # No activation on final dense layer
	tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
	])
	return model
	