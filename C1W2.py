"""
Computer Vision on the Fashion MNIST dataset.

Recognizes types of clothing items
by analizing their photo.

Args:
    None

Returns:
    string: Clothing item class name
"""
# %%
# import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow.keras.layers as layers
# %matplotlib inline
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = 20, 9

# %%
mnist = tf.keras.datasets.fashion_mnist

# %%
# load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
# display size of train and test datasets
print(len(train_labels), len(test_labels))

# %%
# print sample image to test the dataset
# and the numeric values that make up the image
i = 0
plt.imshow(train_images[i])
print(train_labels[i])
print(train_images[i])

# %%
# since the numeric values for the images
# are in the range [0,255] we can normalize them
# by dividing them by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
# creating a simple Fully Conected neural network
# 128 hidden units
# 10 output units, each representing a class
model = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %%
# bundle the model with an optimizer,
# loss function and metrics to evaluate the fit
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
# fit the model to the training data
model.fit(train_images, train_labels, epochs=5)

# %%
# print the evaluation of the model
# print(model.metrics_names)
# print(model.evaluate(test_images, test_labels))
# this looks nicer
pd.Series(model.evaluate(test_images, test_labels),
          index=model.metrics_names, name=128)

# %%
# print a sample prediction
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# %%
# create and train a larger verison on the model
model_512 = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_512.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model_512.fit(train_images, train_labels, epochs=5)

# %%
pd.Series(model_512.evaluate(test_images, test_labels),
          index=model.metrics_names, name=512)

# %%
# training and ever larger model, 1024 hidden units
model_1024 = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_1024.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

model_1024.fit(train_images, train_labels, epochs=5)

# %%
pd.Series(model_1024.evaluate(test_images, test_labels),
          index=model.metrics_names, name=1024)

# %%
model.evaluate(test_images, test_labels)

# %%
# training a version of the model with two hidden layers
model_2_128 = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_2_128.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

model_2_128.fit(train_images, train_labels, epochs=5)

# %%
pd.Series(model_2_128.evaluate(test_images, test_labels),
          index=model.metrics_names, name=2128)

# %%
# training the original model for longer
model_longer = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_longer.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

model_longer.fit(train_images, train_labels, epochs=30)

# %%
pd.Series(model_longer.evaluate(test_images, test_labels),
          index=model.metrics_names, name=30)

# %%
# using a callback to stop the training
# when the model reaches a threshold accuracy
# for this example set to 85%
model_callback = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

callbacks = tf.keras.callbacks.EarlyStopping(monitor='accuracy')

model_callback.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

model_callback.fit(train_images, train_labels,
                   epochs=30,
                   callbacks=callbacks)

# %%
pd.Series(model_callback.evaluate(test_images, test_labels),
          index=model.metrics_names, name=85)


# %%

class AccCallback(tf.keras.callbacks.Callback):
    """
    Accuracy Callback.

    Creates a callback that on epoch end
    checks the accuracy, and stops the trainign
    if it's above 99%.

    Args:
        None

    Returns:
        callback: a tf.keras.callbacks.Callback object
    """

    def on_epoch_end(self, epoch, logs={}):
        """Check the accuracy, and stops the trainign if it's above 99%."""
        if logs.get('accuracy') > 0.99:
            print('\nReached 99% Accuracy\nStopping Training!')
            self.model.stop_training = True


model_callback_2 = tf.keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

callbacks = AccCallback()

model_callback_2.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

model_callback_2.fit(train_images, train_labels,
                     epochs=30,
                     callbacks=callbacks)

# %%
pd.Series(model_callback_2.evaluate(test_images, test_labels),
          index=model.metrics_names, name=85)
