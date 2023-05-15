# ##########_____DOG_EMOTION_RECOGNITION_____##########

# ImportingLibraries

import numpy as np
# import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization, Flatten, MaxPool2D
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

# from keras.layers import Conv2D, Reshape
# from keras.utils import Sequence
# from keras.backend import epsilon
# from sklearn.model_selection import train_test_split

# from keras.layers import GlobalAveragePooling2D
# from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
# from tqdm.notebook import tqdm_notebook as tqdm
import os

# Data preprocessing
print(os.listdir('./images'))
angry = './images/angry/'
happy = './images/happy/'
sad = './images/sad/'

angry_path = os.listdir(angry)
happy_path = os.listdir(happy)
sad_path = os.listdir(sad)


# %%

def load_path(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (224, 224))
	return img[..., ::-1]


# %%
fig = plt.figure(figsize=(10, 10))
for i in range(9):
	plt.subplot(3, 3, i + 1)
	plt.imshow(load_path(angry + angry_path[i]), cmap='gray')
	plt.suptitle("Angry")
	plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(9):
	plt.subplot(3, 3, i + 1)
	plt.imshow(load_path(angry + angry_path[i]), cmap='gray')
	plt.suptitle("Angry Dogs")
	plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(9):
	plt.subplot(3, 3, i + 1)
	plt.imshow(load_path(happy + happy_path[i]), cmap='gray')
	plt.suptitle("Happy Dogs")
	plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(9):
	plt.subplot(3, 3, i + 1)
	plt.imshow(load_path(sad + sad_path[i]), cmap='gray')
	plt.suptitle("Sad Dogs")
	plt.axis('off')

plt.show()
# %%

# Modelling of the data
dataset_path = './images'
data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1. / 255,
                                   validation_split=0.3)

train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=64,
                                          subset="training")
val = data_with_aug.flow_from_directory(dataset_path,
                                        class_mode="binary",
                                        target_size=(96, 96),
                                        batch_size=64,
                                        subset="validation"
                                        )
# %%
# VGG19 Model

vgg19_model = VGG19(include_top=False, weights="imagenet",
                    input_shape=(96, 96, 3))
# %%
vgg19_model.output[-1]
# %%
model = Sequential([vgg19_model,
                    Flatten(),
                    Dense(3, activation="sigmoid")])
model.layers[0].trainable = False
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")
model.summary()


# %%
def scheduler(epoch):
	if epoch <= 2:
		return 0.001
	elif 2 < epoch <= 15:
		return 0.0001
	else:
		return 0.00001


lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
# %%
hist = model.fit(train,
                 epochs=20,
                 callbacks=[lr_callbacks],
                 validation_data=val)

# %%

epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])

model.save_weights('./model_weights.h5')

# %%
predictions = model.predict(val)

# %%

val_path = "./images/"

plt.figure(figsize=(10, 10))

start_index = 250

for i in range(16):
	plt.subplot(4, 4, i + 1)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	pred = np.argmax(predictions[[start_index + i]])

	gt = val.filenames[start_index + i][9:13]

	if gt == "angry":
		gt = 0
	elif gt == 'happy':
		gt = 1
	else:
		gt = 2

	if pred != gt:
		col = "r"
	else:
		col = "g"

	if gt == "angry":
		gt = 'angry'
	elif gt == 'happy':
		gt = 'happy'
	else:
		gt = 'sad'

	plt.xlabel('i={}, pred={}, gt={}'.format(start_index + i, pred, gt), color=col)
	plt.imshow(load_path(val_path + val.filenames[start_index + i]))
	plt.tight_layout()

plt.show()

# %%
from flask import Flask, request, jsonify
from PIL import Image

# Define the Flask app
app = Flask(__name__)


# API endpoint for classifying dog emotions
@app.route('/classify', methods=['POST'])
def classify_emotion():
	# Check if an image file was included in the request
	if 'image' not in request.files:
		return jsonify({'error': 'No image file provided'})

	# Read the image file and preprocess it
	image_file = request.files['image']
	image = Image.open(image_file)
	image = image.resize((96, 96))  # Resize to match the input size expected by the model
	image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
	image = np.expand_dims(image, axis=0)  # Add a batch dimension

	# Make predictions using the model
	predictions = model.predict(image)
	predicted_class = np.argmax(predictions[0])
	emotion_labels = ['angry', 'happy', 'sad']
	predicted_emotion = emotion_labels[predicted_class]

	# Return the classification result as a JSON response
	return jsonify({'emotion': predicted_emotion})


# Run the Flask app
if __name__ == '__main__':
	app.run()

