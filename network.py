##############################################
# Author: Jakub Svoboda
# Email:  xsvobo0z@stud.fit.vutbr.cz
# School: Brno University of Technology
##############################################
# This code handles the creation of the neural network and loades its weights.
# The network can then create embeddings from image data. The alignment of the face
# is realized by MTCNN module for python.
##############################################


from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
import os


IMG_SIZE = 224							# 224 for mobilenet, 299 for inception
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BACKBONE = 'MobileNetV3Large'			# Selects the backbone CNN, see getNetwork() for all optinons
WEIGHTS = os.path.join('checkpoints' ,'mobile_9708.hdf5')


# Converts the 'img' passed to tensorflow float type and resizes it to appropriate NN input size.
def processImage(img):
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))
	return img	# Return as TF tensor


# Creates a network based on passed parameters. The 'backbone' is the CNN for feature extraction,
# 'embeddingSize' is the output dimension (128 is default, larger values generally do not inprove accuracy).
# 'fcSize' specifies the size of the intermediate fully connected layer.
# 'l2Norm' boolean parameter adds l2 normalization lambda layer at the very end of the network.
def getNetwork(backbone = 'ResNet50V2', embeddingSize=128, fcSize=1024, l2Norm=True):
	if backbone == 'ResNet50V2':
		baseModel = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'ResNet101V2':
		baseModel = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'ResNet152V2':
		baseModel = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'InceptionV3':
		baseModel = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'MobileNetV3Large':
		baseModel = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'MobileNetV3Small':
		baseModel = tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights=None)	
	elif backbone == 'DenseNet169':
		baseModel = tf.keras.applications.DenseNet169(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'DenseNet121':
		baseModel = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights=None)
	elif backbone == 'InceptionResNetV2':
		baseModel = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights=None)
	else:
		raise Exception("Backbone network not matched to any known architecture:", backbone)

	baseModel.trainable = True 						# This code should only be used for inference (see the training script)
	model = tf.keras.Sequential([
	baseModel,											
	tf.keras.layers.GlobalAveragePooling2D(),			# Global average pooling pools across both image dimensions
	tf.keras.layers.Dense(fcSize, activation='relu'),	# Dense layer of size 'fcSize' with ReLU activation
	tf.keras.layers.BatchNormalization(),				
	tf.keras.layers.Dense(embeddingSize), ])			# Output dimensions specified by 'embeddingSize'
	
	if l2Norm:											# Optionally (though recomended), normalize by L2 normalization
		model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

	return model


# Class that encapsulates the identification network as well as the detector (MTCNN)
# For inference, run the detectFace() method
class IdentityNetwork():
	
	# Constructor
	def __init__(self):
		# Initialilze MTCNN localiator
		self.detector = MTCNN(steps_threshold=[0.70,0.70,0.8])
		# MARGIN pixels from each direction of the bounding box should still be inculded in the cutout
		self.margin = 32
		self.initModel()


	# Creates the model and pulls the stored weights from disk	
	def initModel(self):
		self.model = getNetwork(backbone = BACKBONE, embeddingSize=128, fcSize=1024, l2Norm=True)
		self.model.summary()							# Prints model architecture to console
		self.model.load_weights(WEIGHTS)				# You can point this to your own model
		self.model.trainable = False


	# Localizes face in passed numpy 'img' and generates embedding through the network.
	def detectFaces(self, img):
		# Get the face bounding box for given image
		result = self.detector.detect_faces(img)
		# There can be zero faces in the image or mtcnn can fail at detection, in these cases, return empty array
		if len(result) <= 0:
			print('MTCNN found 0 faces')
			return None
		# Otherwise for each face detected, do a crop	
		boxes = []
		for res in result:
			bb = result[0]['box']
			x = np.max(bb[0]-(self.margin//2),0)
			y = np.max(bb[1]-(self.margin//2),0) 
			x2 = np.minimum(bb[0]+bb[2]+(self.margin//2),img.shape[0])
			y2 = np.minimum(bb[1]+bb[3]+(self.margin//2),img.shape[1])
			img = img[y:y2, x:x2]
			boxes.append(img)

		# Resize each found face to input size
		images = []
		for b in boxes:
			image = processImage(b)
			images.append(image)
		images = tf.convert_to_tensor(images)

		# Create embedding from each image
		embeddings = self.model.predict(images)
		
		return embeddings





