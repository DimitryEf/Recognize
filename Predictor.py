from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tensorflow.keras.preprocessing import image


def plot_image_single(predictions_array, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  color = 'black'
  plt.xlabel("{} {:2.0f}% ".format(class_names[predicted_label],
                                100*np.max(predictions_array)),
                                color=color)


if __name__ == "__main__":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights('./checkpoints/my_checkpoint')

    #

    img_path = "2.jpeg"
    img = image.load_img(img_path)
    image_x = 28
    image_y = 28
    img = img.resize((image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y))
    img = img / 255.0
    pred_probab = model.predict(img)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    print(class_names[pred_class], str(int(round(100*max(pred_probab))))+"%")

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image_single(pred_probab, image.load_img(img_path))
    plt.show()



