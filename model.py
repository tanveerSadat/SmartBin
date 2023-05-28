# importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow import keras

# setting the path to the garbage classification photos
train_dir = os.path.join('insert path to dataset') # linked in github - README.md file
labels = ['garbage', 'organics', 'recycling']

# define the image size and batch size
image_size = (256, 256)
batch_size = 32

# create the image dataset from the directory
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  
  train_dir,
  labels='inferred',
  label_mode='categorical',
  class_names=labels,
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=image_size,
  batch_size=batch_size
)

# split the dataset into seperate datasets for training and validation 
validation_split = 0.2
num_validation_samples = int(validation_split * image_dataset.cardinality().numpy())

train_dataset = image_dataset.skip(num_validation_samples)
validation_dataset = image_dataset.take(num_validation_samples)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# display information about the datasets
print("\nNumber of Training Samples:", image_dataset.cardinality().numpy() - num_validation_samples)
print("Number of Validation Samples:", num_validation_samples)
print("Number of Classes:", len(labels))
print("\n")

# normalizes data
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# creates a CNN to process images
model = tf.keras.Sequential([

  Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
  MaxPooling2D(2, 2),
  Dropout(0.2),

  Conv2D(32, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Dropout(0.2),

  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Dropout(0.2),

  Flatten(),
  Dense(64, activation='relu'),
  Dropout(0.4),
  Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = 0.0001), metrics=['accuracy'])

# train the model
model.fit(
  train_dataset,
  epochs=12,
  batch_size=36,
  validation_data=validation_dataset
)

# set a model path and save the model 
model_path = 'create a path to save the model to' 
model.save(model_path)
print("Model saved successfully.")
