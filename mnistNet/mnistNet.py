import io
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

import vggModel

def _normalize_img(img, label):
	img = tf.cast(img, tf.float32) / 255.
	return (img, label)

train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)


train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.map(_normalize_img)

test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.map(_normalize_img)


model = vggModel.getModel()


# Compile the model
model.compile(
	optimizer=tf.keras.optimizers.Adam(0.001),
	loss=tfa.losses.TripletSemiHardLoss())

# Train the network
history = model.fit(train_dataset, epochs=1)

# Evaluate the network
results = model.predict(test_dataset)


# Save test embeddings for visualization in projector
np.savetxt("vecs.tsv", results, delimiter='\t')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for img, labels in tfds.as_numpy(test_dataset):
	[out_m.write(str(x) + "\n") for x in labels]
out_m.close()



#try:
#  from google.colab import files
#  files.download('vecs.tsv')
#  files.download('meta.tsv')
#except:
#  pass



