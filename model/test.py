# importing libraries
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# load the saved model
model = keras.models.load_model('insert the path to the saved model')

# load and preprocess the image
image_path = 'insert a path to an image to test model'
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# predict accuracy based on given image
predictions = model.predict(image)
print(predictions)

item_labels = ['garbage', 'organics', 'recycling']

predicted_label = np.argmax(predictions)
predicted_class = item_labels[predicted_label].title()
confidence = predictions[0, predicted_label]

# print the prediction
print(f"\nPredicted Class: {predicted_class}")
print(f"Arr: {predictions}") # predictions array
print(f"Confidence: {confidence}\n")
