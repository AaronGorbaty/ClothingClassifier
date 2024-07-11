# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

# Load the dataset from the local file
path = './mnist.npz'
with np.load(path) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()
# loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

#configure and compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#Use the Model.fit method to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)

#evaluate accuracy using a validation set
model.evaluate(x_test,  y_test, verbose=2)

#returning probability
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])