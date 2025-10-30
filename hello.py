import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

#getting the roof dataset 
data_dir = pathlib.Path(r"C:\Users\lsboyin\OneDrive - IL State University\StartingProject\PRJ-4671\PRJ-4671\Project--roof-subassembly-damage-detection-image-datasets\data\Global Damage Classifier data\train data")

#showing the number of images in the dataset
image_count = len(list(data_dir.glob('*/*')))
print(f"Total Images found: {image_count}")
if(image_count == 0):
    print("Warning: No images found. Check your dataset path and format.")
    all_files = list(data_dir.glob('*/*'))
    print(f"Found Files (first 5): {[str(f) for f in all_files[:5]]}")

#not really working right now
destroyed = list(data_dir.glob('Destroyed/*'))
PIL.Image.open(str(destroyed[0]))

#training 80% of the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size=(180,180),
    batch_size=32
)

#validating with 20% of the dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="validation",
    seed=123,
    image_size=(180,180),
    batch_size=32
)

#shows the different classifications (Destroyed, Not Destroyed, Partial Destroyed)
class_names = train_ds.class_names
print(class_names)

#shows us a couple of images through matplotlib and classifies what type of image it is 
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    if images.shape[0] < 9:
        print(f"Warning: Batch size is {images.shape[0]}, less than 9. Displaying available images.")
        num_to_display = images.shape[0]
    else:
        num_to_display = 9
    for i in range(num_to_display):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

num_classes = len(class_names) 
model = Sequential([ 
    #Rescales images to [0,1] and sets input image size
	layers.Rescaling(1./255, input_shape=(180,180, 3)), 
    # Adds a convolutional layer with 16 filters and ReLU activation(dot product)
	layers.Conv2D(16, 3, padding='same', activation='relu'), 
    #Adds a max-pooling layer to down sample feature maps and decreases parameters
	layers.MaxPooling2D(), 
	layers.Conv2D(32, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
	layers.Conv2D(64, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
	layers.Flatten(), 
	layers.Dense(128, activation='relu'), 
	layers.Dense(num_classes) 
]) 

#uses adam optimizer and using CategoricalCrossEntropy to find minimum loss during epochs 
model.compile(optimizer='adam', 
			loss=tf.keras.losses.SparseCategoricalCrossentropy( 
				from_logits=True), 
			metrics=['accuracy']) 
model.summary() 
