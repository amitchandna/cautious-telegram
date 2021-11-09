from PIL import Image
import os, os.path
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Load in the data from the photo sets above - this will be used for augmentation in a moment...
malaria = []
not_malaria = []
malaria_path = "/Users/amitchandna/Documents/Data_Science/malaria_identification/malaria_image_dataset/malaria"
not_malaria_path = "/Users/amitchandna/Documents/Data_Science/malaria_identification/malaria_image_dataset/no_malaria"
valid_images = [".jpg",".gif",".png",".tga"]
#Define a function to load in the data..
def load_image_data(path,array):
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        array.append(Image.open(os.path.join(path, f)))
#Call on the function and load in the data..
load_image_data(malaria_path,malaria)
load_image_data(not_malaria_path,not_malaria)

#Start to build a net to augment the data in either malaria or not_malaria
data_augmentation = tf.keras.Sequential([
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(5)])

image = tf.expand_dims(malaria[1], 0)
plt.figure(figsize=(10, 10))
#Run this augmentor X number of times.
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
  plt.show()
#Augment the data in as many ways as possible and bring the final number of data points of both yes and no to about 45,000 each
#This should then result in about 90k data points with which a Neural Net can be built/trained.

#Neural Net #1 for Augmentation
