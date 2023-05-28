# importing libraries
import cv2
import numpy as np 
from tensorflow import keras

# find path to model
model = keras.models.load_model('insert the path to the saved model', compile=False) 
image_size = (256, 256) # define image size
class_labels = ['garbage', 'organics', 'recycling']

# function preprocess_image resizes the image and converts it to grey scale 
def preprocess_image(image):
    
  image = cv2.resize(image, image_size)
  image = image.astype(np.float32)
  image /= 255.0
  image = np.expand_dims(image, axis=0)

  return image

# define a video capture object
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

if not cap.isOpened():
  raise IOError("Cannot open webcam")

while(True):

  ret, frame = cap.read()
  preprocessed_frame = preprocess_image(frame)
  predictions = model.predict(preprocessed_frame)
  predicted_label = np.argmax(predictions)

  label = class_labels[predicted_label]  

  cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  cv2.imshow('Object Classification', frame)

# define a button to quit the camera - 'q' 
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()