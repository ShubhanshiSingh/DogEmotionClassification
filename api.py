#%%
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

#%%

# Load the trained model
model = load_model('C:/Users/shubh/DogEmotionClassification/model_weights.h5')

# Define the Flask app
app = Flask(__name__)


# API endpoint for classifying dog emotions
@app.route('/classify', methods=['POST'])
def classify_emotion():
	# Check if an image file was included in the request
	if 'image' not in request.files:
		return jsonify({'error': 'No image file provided'})

	# Read the image file and preprocess it
	image_file = request.files['./small-fuffy-dog-breeds-1623362663.jpg']
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
