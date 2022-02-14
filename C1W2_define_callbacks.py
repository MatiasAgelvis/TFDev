import tensorflow as tf

# %%


class AccCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('\nReached 99% Accuracy\nStopping Training!')
            self.model.stop_training = True


# %%
mnist = tf.keras.datasets.mnist

# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

# %%

callbacks = AccCallback()

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
