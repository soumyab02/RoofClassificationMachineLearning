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
from keras.models import Sequential
from keras.callbacks import Callback
import pathlib
import settings1 as settings

#getting the roof dataset 
data_dir = pathlib.Path(r"C:\Users\lsboyin\OneDrive - IL State University\StartingProject\PRJ-4671\PRJ-4671\Project--roof-subassembly-damage-detection-image-datasets\data\Global Damage Classifier data\train data")
AUTOTUNE = tf.data.AUTOTUNE

#showing the number of images in the dataset
image_count = len(list(data_dir.glob('*/*')))
print(f"Total Images found: {image_count}")
if(image_count == 0):
    print("Warning: No images found. Check your dataset path and format.")
    all_files = list(data_dir.glob('*/*'))
    print(f"Found Files (first 5): {[str(f) for f in all_files[:5]]}")

#training 80% of the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size=(settings.pixel_size,settings.pixel_size),
    batch_size=64
)

#validating with 20% of the dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="validation",
    seed=123,
    image_size=(settings.pixel_size,settings.pixel_size),
    batch_size=64
)

#shows the different classifications (Destroyed, Not Destroyed, Partial Destroyed)
class_names = train_ds.class_names
print(class_names)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(settings.pic_randomization), layers.RandomZoom(settings.pic_randomization), layers.RandomTranslation(settings.pic_translation, settings.pic_translation)])
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    if images.shape[0] < 9:
        print(f"Warning: Batch size is {images.shape[0]}, less than 9. Displaying available images.")
        num_to_display = images.shape[0]
    else:
        num_to_display = 9
    for i in range(num_to_display):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

#3 num_classes
num_classes = len(class_names) 
model = Sequential([ 
    data_augmentation,
	layers.Rescaling(1./255, input_shape=(settings.pixel_size,settings.pixel_size, 3)), 
    #outputs a 3D tensor where depth corresponds to the number of filters
    #It enables the model to detect local patterns like edges or textures
    #specialized for processing grid-like data such as images
	layers.Conv2D(16, 3, padding='same', activation='relu'), 
    #Pooling layers downsample feature maps reducing spatial dimensions while retaining important features
    #They help control overfitting and decrease computation
	layers.MaxPooling2D(), 
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(), 
	layers.Conv2D(64, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
    #reshapes a multi-dimensional tensor into a one-dimensional vector
    #This is important before passing data from convolutional layers to fully connected layers
	layers.Flatten(),
	#(128 + 1) * num_classes params
    #fully connected layer where every input is connected to every neuron in the layer
    #It is most commonly used at the end of convolutional networks or in feedforward architectures
	layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
	layers.Dense(num_classes, activation='softmax')
]) 

#uses adam optimizer and using CategoricalCrossEntropy to find minimum loss during epochs 
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy']) 

class LogEveryNEpochs(Callback):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def on_epoch_end(self, epoch, logs=None):
        # Epoch numbers are 0-indexed, so we add 1 for display purposes
        if (epoch + 1) % self.n == 0:
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']} - "
                  f"Loss: {logs.get('loss'):.4f} - Acc: {logs.get('accuracy'):.4f} - "
                  f"Val Loss: {logs.get('val_loss'):.4f} - Val Acc: {logs.get('val_accuracy'):.4f}")
            
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
reduceLR_cb = keras.callbacks.ReduceLROnPlateau(montior='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0)

# Instantiate the custom callback to log every 10 epochs
log_callback = LogEveryNEpochs(10)
callbacks = [earlystop_cb, reduceLR_cb, log_callback]
epochs = settings.epochs
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,verbose=0)

#visualizing the accuracy and loss
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
epochs_range = range(len(val_acc))
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