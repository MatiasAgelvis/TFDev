"""
Classify the Dogs vs. Cats dataset.

Explore the example data of Dogs vs. Cats
Build and train a neural network to classify between the two pets
Evaluate the training and validation accuracy
Prevent Overfitting
"""

# %%
!wget -P ./datasets --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

# %%
import zipfile

local_zip = './datasets/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(path='./datasets')

zip_ref.close()

# %%
import os

base_dir = './datasets/cats_and_dogs_filtered'

print('Contents of base directory')
print(os.listdir(base_dir))

print('\nContents of base directory')
print(os.listdir(os.path.join(base_dir, 'train')))

print('\nContents of base directory')
print(os.listdir(os.path.join(base_dir, 'validation')))

# %%
train_dir = os.path.join(base_dir, 'train')
validation_dir  = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# %%
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

# %%
print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))

print('total training cat images :', len(os.listdir(validation_cats_dir)))
print('total training dog images :', len(os.listdir(validation_dogs_dir)))

# %%
# preview dataset
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows = 4
ncols = 4

pic_index = 0

# %%
fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)

next_cat_pix = [ os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index:pic_index+8] ]

next_dog_pix = [ os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index:pic_index+8] ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# %%
# model definition
import tensorflow as tf
import tensorflow.keras.layers as Klayers

data_shape = (150, 150, 3)

model = tf.keras.models.Sequential([
        Klayers.Input(data_shape),

        Klayers.Conv2D(16, (3,3), activation='relu'),
        Klayers.MaxPool2D(2,2),

        Klayers.Conv2D(32, (3,3), activation='relu'),
        Klayers.MaxPool2D(2,2),

        Klayers.Conv2D(64, (3,3), activation='relu'),
        Klayers.MaxPool2D(2,2),

        Klayers.Flatten(),

        Klayers.Dense(512, activation='relu'),
        Klayers.Dense(1, activation='sigmoid')
])

# %%
model.summary()


# %%
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% md

## Data Preprocessing
# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255)
validation_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))

# %%
# model training
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

# %%
# visualizing intermediate representations
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# define a new model taht outputs the intermediate states of each layer
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255

succesive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, succesive_feature_maps):
    # for the convolution-pooling layers
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]

        display_grid = np.zeros((size, size * n_features))

        # normalize the feature maps to be in the range [0, 255]
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


# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.title('Training and validation accuracy')
plt.show()

plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title('Training and validation loss')
plt.show()

# %% md
## The model is overfitted!
