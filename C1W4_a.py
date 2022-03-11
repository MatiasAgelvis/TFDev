"""
Training with ImageDataGenerator.

Builds a model on the Human or Horses dataset,
 using ImageDataGenerator to prepare and augment the dataset.
"""
# %%
# Download the dataset (about 142MB)
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O ./horse-or-human.zip

# %%
# unzip the downloaded images
import os
import zipfile

local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./horse-or-human')
zip_ref.close()

# %%
# check that each of the directories
train_horses_dir = os.path.normpath('./horse-or-human/horses')
train_humans_dir = os.path.normpath('./horse-or-human/humans')

train_horses_names = os.listdir(train_horses_dir)
train_humans_names = os.listdir(train_humans_dir)

print('Some files from the horses directory')
print(train_horses_names[:10])
print('Some files from the humans directory')
print(train_humans_names[:10])

print('total training horse images:', len(train_horses_names))
print('total training human images:', len(train_humans_names))

# %%
# setup example plots
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0

# %%
# display a nrows by ncols grid of images starting with the pic_index image

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

next_horse_pic = [os.path.join(train_horses_dir, fname)
                  for fname in train_horses_names[pic_index:pic_index + (nrows * ncols // 2)]]

next_human_pic = [os.path.join(train_humans_dir, fname)
                  for fname in train_humans_names[pic_index:pic_index + (nrows * ncols // 2)]]

for i, img_path in enumerate(next_horse_pic + next_human_pic):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# %% md
# Build the model
# %%
import tensorflow as tf
import  tensorflow.keras.layers as layers

model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300,300,3)),
    layers.MaxPool2D(2, 2),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),

    # a [0,1] output for the binary claisification problem of 'horse or human'
    layers.Dense(1, activation='sigmoid')
])

# %%
model.summary()

# %%
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

# %% md
# Data Prepocessing
# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# divides all color values by 255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    './horse-or-human/',      # Directory containing the images
    target_size=(300, 300),    # resizes the images
    batch_size=128,
    class_mode='binary'
)

# %%
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1
)

# %% md
# Model prediction
# %%
import numpy as np

def model_prediction(path, model):

    # image loading
    img = tf.keras.preprocessing.image.load_img(path, target_size(300,300))
    x = tf.keras.preprocessing.image.img_to_array(img)
    # normalization
    x = x / 255
    # creates a batch of images with a single image
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # check why this is necesary
    print('x.shape', x.shape)
    print('images.shape', images.shape)

    classes = model.predict(images)
    print(classes[0])

    if classes[0] > 0.5:
        print('It\'s a Human')
    else:
        print('It\'s a Horse')

# %% md
# Vizualizing Intermediate Representations

# %%
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# define a new model that takes an image as input,
# and outputs each intermediate state
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

horse_img_files = [os.path.join(train_horses_dir, f) for f in train_horses_names]
human_img_files = [os.path.join(train_humans_dir, f) for f in train_humans_names]
img_path = random.choice(horse_img_files + human_img_files)


img = load_img(img_path, target_size=(300, 300))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers[1:]]

# display the Representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:

        n_features = feature_map.shape[-1]

        size = feature_map.shape[1]

        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')

            display_grid[:, i * size : (i + 1) * size] = x

        scale = 20 / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
