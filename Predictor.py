from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PIL import Image


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')


if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    # arr2im = Image.fromarray(test_images[0])
    # arr2im.save('1.jpeg')

    print(type(test_images))
    print(len(test_images))
    print(type(test_images[0]))

    new_test_images = np.array([np.array(Image.open("1.jpeg"))])

    print(type(new_test_images))
    print(type(new_test_images[0]))

    train_images.shape
    # Each Label is between 0-9
    train_labels
    new_test_images.shape

    # If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255.


    new_test_images = new_test_images / 255.0


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights('./checkpoints/my_checkpoint')

    # test_loss, test_acc = model.evaluate(new_test_images, test_labels)
    # print('Test accuracy:', test_acc)



    predictions = model.predict(new_test_images)[0]
    predictions

    np.argmax(predictions)
    # Model is most confident that it's an ankle boot. Let's see if it's correct

    test_labels[0]

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()
