from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 244
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BACKBONE = 'MobileNetV3Small'

def processImage(img):
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, (IMG_SIZE,IMG_SIZE))
	return img

def getNetwork(backbone = 'ResNet50V2', embeddingSize=128, fcSize=1024, l2_norm=True):
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

	baseModel.trainable = True
	model = tf.keras.Sequential([
	baseModel,
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(fcSize, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dense(embeddingSize), ])
	
	if l2_norm:
		model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

	return model

class IdentityNetwork():
	def __init__(self):
		# Initialilze MTCNN localiator
		self.detector = MTCNN(steps_threshold=[0.70,0.70,0.8])
		# MARGIN pixels from each direction of the bounding box should still be inculded in the cutout
		self.margin = 32
		self.initModel()

	def initModel(self):
		self.model = getNetwork(backbone = BACKBONE, embeddingSize=128, fcSize=1024, l2_norm=True)
		self.model.summary()
		self.model.load_weights('./checkpoints/best_mobile_964.hdf5')

	def detectFaces(self, img):
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





