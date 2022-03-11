"""
Handling Complex Images - Happy or Sad Dataset

The happy or sad dataset, contains 80 images of emoji-like faces,
40 happy and 40 sad.
"""

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# %%
import zipfile

!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "./datasets/happy-or-sad.zip"

# %%

zip_ref = zipfile.ZipFile("./datasets/happy-or-sad.zip", 'r')
zip_ref.extractall("./datasets/happy-or-sad")
zip_ref.close()

# %%
from tensorflow.keras.preprocessing.image import load_img
happy_dir = './datasets/happy-or-sad/happy/'
sad_dir = './datasets/happy-or-sad/sad/'

print('Sample happy image')
plt.imshow(load_img(
    f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
plt.show()

print('Sample sad image')
plt.imshow(load_img(
    f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
plt.show()

# %%
from tensorflow.keras.preprocessing.image import img_to_array

sample_image = load_img(
    f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")

sample_array = img_to_array(sample_image)

print('Each image has shape:', sample_array.shape)

print('The maximum value used is:', np.max(sample_array))
data_shape = sample_array.shape
# %% md
The images are 150x150 and have 3 color channels (RGB)

# %%
class accCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None \
        and logs.get('accuracy') > 0.999:
            self.model.stop_training = True

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_generator():
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory='./datasets/happy-or-sad',
        target_size=(150,150),
        batch_size=10,
        class_mode='binary'
    )

    return train_generator

# %%
gen = image_generator()

# %%
from tensorflow.keras import optimizers, losses
import tensorflow.keras.layers as Klayers

def train_happy_sad_model(train_generator, input_shape):

    callbacks = accCallback()


    model = tf.keras.models.Sequential([
        Klayers.Input(input_shape),

        Klayers.Conv2D(2^7, 3, activation='relu'),
        Klayers.MaxPool2D(2),

        Klayers.Conv2D(2^6, 3, activation='relu'),
        Klayers.MaxPool2D(2),

        Klayers.Conv2D(2^5, 3, activation='relu'),
        Klayers.MaxPool2D(2),

        Klayers.Flatten(),
        Klayers.Dense(2^7, activation='relu'),
        # output layer
        Klayers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x=train_generator,
                        epochs=20,
                        callbacks=[callbacks])

    return history

# %%
history = train_happy_sad_model(image_generator(), data_shape)

# %%
print('The model achived the desired accuracy of 99.9% after',
       len(history.epoch),
       'epochs')
# %%
history.history
