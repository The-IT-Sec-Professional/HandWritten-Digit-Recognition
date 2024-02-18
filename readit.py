import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Added code

# end


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

loss, accuracy= model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

model.save('digits.model')

model = tf.keras.models.load_model('digits.model')

img = cv2.imread('digit.png')[:,:,0]
img = np.invert(np.array([img]))

prediction =model.predict(img)
print("Prediction:{}".format(np.argmax(prediction)))
plt.imshow(img[0])
plt.show()
