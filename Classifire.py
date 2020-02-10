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

    # print(train_images[0])
    # print(train_labels[0])
    # print(test_images[0])
    # print(test_labels[0])

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    train_images.shape
    # Each Label is between 0-9
    train_labels
    test_images.shape

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(test_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    # If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255.

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if False:
        start_time = timer()
        model.fit(train_images, train_labels, epochs=10)
        print("Total training time: {:g} secs".format(timer() - start_time))
        model.save_weights('./checkpoints/my_checkpoint')
    else:
        model.load_weights('./checkpoints/my_checkpoint')

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    predictions[0]
    print(predictions[0])

    np.argmax(predictions[0])
    # Model is most confident that it's an ankle boot. Let's see if it's correct

    test_labels[0]

    i = 100
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

    #
    # img1 = image.load_img("2.jpeg")
    # test_image1 = np.reshape(test_images[5], (-1, 28, 28))
    # pred1 = model.predict(test_image1)[0]
    # print(pred1)
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image_single(pred1, test_images[5])
    # plt.show()

    #

    img = image.load_img("1.jpeg")
    image_x = 28
    image_y = 28
    img = img.resize((image_x, image_y))
    # img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y))
    img = img / 255.0
    pred_probab = model.predict(img)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    print("***")
    print(pred_probab)
    print(max(pred_probab), pred_class)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image_single(pred_probab, image.load_img("1.jpeg"))
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions, test_labels)
    plt.show()



