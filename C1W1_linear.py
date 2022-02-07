"""
Simple Linear Regression with TensorFlow.

Does a Linear Regression fitting a single perceptron
to the function y=2x-1 for 300 epochs.

Args:
    None

Returns:
    int: 18.879307
"""
# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas
import matplotlib.pyplot as plt
%matplotlib inline
# %%
# Simple Linear Regression
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
# %%
# data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# %%
history = model.fit(xs, ys, epochs=300)
# %%
print(model.predict([10.0]))

# %%
pandas.Series(history.history['loss']).plot()
