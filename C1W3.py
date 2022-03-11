"""
Basics of convolutions.

First its a demonstration of how convolutions work,
then their application in deep learning models.
"""

# %% md
# Section 1

How convolutions work

# %%
import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# %%
# the image
i = misc.ascent()

# plot configs
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

# %%
# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

# Experiment with different values for fun effects.
# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# filter = [ [-1, -1, 2], [-1, 2, -1], [2, -1, -1]]
# filter = [ [2, 1, 2], [1, -2, -1], [0, -1, 2]]

# If all the digits in the filter don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1
# if you want to normalize them
weight = 1

# %%
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

print('(size_x, size_y)\n', (size_x, size_y))

# %%
# "The Convolution"
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0
        convolution += i[x-1, y-1] * filter[0][0]
        convolution += i[x,   y-1] * filter[0][1]
        convolution += i[x+1, y-1] * filter[0][2]
        convolution += i[x-1, y  ] * filter[1][0]
        convolution += i[x,   y  ] * filter[1][1]
        convolution += i[x+1, y  ] * filter[1][2]
        convolution += i[x-1, y+1] * filter[2][0]
        convolution += i[x,   y+1] * filter[2][1]
        convolution += i[x+1, y+1] * filter[2][2]

        convolution *= weight

        convolution = max(convolution, 0)
        convolution = min(convolution, 255)

        i_transformed[x, y] = convolution

# %%
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
# plt.axis('off')
plt.show()
# notice that image size is preserved

# %%
# now appliyin a (2,2) pooling
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x,new_y))

for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x+1, y])
        pixels.append(i_transformed[x, y+1])
        pixels.append(i_transformed[x+1, y+1])
        newImage[int(x/2), int(y/2)] = max(pixels)

plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()
# the image is now 265x265

# %% md
# Section 2

A model with convolutions

# %%
import tensorflow as tf

# %%
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

# %%
print(train_images.shape)
print(test_images.shape)

# append a new dimention
# basically turns the array into an image
# really, turns the values into single value arrays
# in TF the final dimension contains the color channels
# new the arrys can be interpreted as a single color channel image
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

print(train_images.shape)
print(test_images.shape)
# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.summary()

# %%
model.fit(train_images, train_labels, epochs=10,
          validation_split=0.2,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=1, verbose=1)])


# %%
test_loss = model.evaluate(test_images, test_labels)
# %%
model.save('fashion_mnist_conv.h5')

# %% md
## Visualizing the convolutions and pooling

# %%
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
img_1 = 0
img_2 = 1
img_3 = 2
conv_num = 3

from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
print(len(layer_outputs))
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

for x in range(0,4):
    f1 = activation_model.predict(tf.reshape(test_images[img_1], (1,28,28,1)))[x]
    axarr[0,x].imshow(f1[0, :, :, conv_num], cmap='inferno')
    axarr[0,x].grid(False)

    f2 = activation_model.predict(tf.reshape(test_images[img_2], (1,28,28,1)))[x]
    axarr[1,x].imshow(f2[0, :, :, conv_num], cmap='inferno')
    axarr[1,x].grid(False)

    f3 = activation_model.predict(tf.reshape(test_images[img_3], (1,28,28,1)))[x]
    axarr[2,x].imshow(f1[0, :, :, conv_num], cmap='inferno')
    axarr[2,x].grid(False)

# %% md
## Exercises
1. Remove filters to the model
1. Remove even more filters to the model
1. Add a layer to the model
1. Remove a layer from the model
1. Implement a Callback for accuracy

# %% md
#### model layers: 2, convolutions: 16

# %%
model_l2_c16 = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((28,28,1)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model_l2_c16.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model_l2_c16.fit(train_images, train_labels, epochs=10)

# %%
model_l2_c16.evaluate(test_images, test_labels)

# %% md
#### model layers: 2, convolutions: 32

# %%
model_l2_c32 = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model_l2_c32.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model_l2_c32.fit(train_images, train_labels, epochs=10)
# %%
model_l2_c32.evaluate(test_images, test_labels)

# %% md
#### model layers: 3, convolutions: 64

# %%
model_l3_c64 = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model_l3_c64.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model_l3_c64.fit(train_images, train_labels, epochs=10)

# %%
model_l2_c32.evaluate(test_images, test_labels)

# %% md
#### model layers: 1, convolutions: 64

# %%
model_l1_c64 = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model_l1_c64.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model_l1_c64.fit(train_images, train_labels, epochs=10)

# %%
model_l1_c64.evaluate(test_images, test_labels)

# %% md
#### model with accuracy callback and more convolutions
#### model layers: 3, convolutions: 128

# %%
class AccCallback(tf.keras.callbacks.Callback):
    """
    Accuracy Callback.

    Creates a callback that on epoch end
    checks the accuracy, and stops the trainign
    if it's above the given accuracy.

    Args:
        None

    Returns:
        callback: a tf.keras.callbacks.Callback object
    """
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs={}):
        """Check the accuracy, and stops the trainign if it's above 99%."""
        if logs.get('accuracy') > self.accuracy:
            print('\nReached 99% Accuracy\nStopping Training!')
            self.model.stop_training = True

# %%
model_l3_c128 = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((28,28,1)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model_l3_c128.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
callback = AccCallback(0.97)
model_l3_c128.fit(train_images, train_labels, epochs=100, callbacks=[callback])

# %%
model_l3_c128.evaluate(test_images, test_labels)
