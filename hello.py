import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import pathlib
from tensorflow import keras
from keras import layers
from keras import layers
from keras.models import Sequential
from keras.callbacks import Callback
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
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) 
#model.summary() 

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
'''class LogEveryNEpochs(Callback):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            print(f"Epoch {epoch + 1}/{self.params['epochs']} - "
                  f"Loss: {logs['loss']:.4f} - "
                  f"Val Loss: {logs['val_loss']:.4f}")

# Instantiate the custom callback
epochs_to_log = 10
custom_logger = LogEveryNEpochs(epochs_to_log)

epochs = 100
# Update the model.fit() call
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[custom_logger])

'''

#visualizing the accuracy and loss
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
epochs_range = range(epochs) 
plt.figure(figsize=(8, 8)) 
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right') 
plt.title('Training and Validation Accuracy') 
plt.subplot(1, 2, 2) 
plt.plot(epochs_range, loss, label='Training Loss') 
plt.plot(epochs_range, val_loss, label='Validation Loss') 
plt.legend(loc='upper right') 
plt.title('Training and Validation Loss') 
plt.show() 