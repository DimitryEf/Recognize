from __future__ import absolute_import, division, print_function

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse as urlparse
from secrets import token_urlsafe
from urllib.parse import parse_qs

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

from PIL import Image
from base64 import decodebytes

import base64
from PIL import Image, ImageFile
from io import BytesIO

import datetime

#  http://localhost:8000/abc?def=/9j/4AAQSkZJRgABAQEAYABgAAD/4QBmRXhpZgAATU0AKgAAAAgABAEaAAUAAAABAAAAPgEbAAUAAAABAAAARgEoAAMAAAABAAIAAAExAAIAAAAQAAAATgAAAAAAAABgAAAAAQAAAGAAAAABcGFpbnQubmV0IDQuMS42AP/bAEMAAgEBAgEBAgICAgICAgIDBQMDAwMDBgQEAwUHBgcHBwYHBwgJCwkICAoIBwcKDQoKCwwMDAwHCQ4PDQwOCwwMDP/bAEMBAgICAwMDBgMDBgwIBwgMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDP/AABEIABwAHAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2gAMAwEAAhEDEQA/AP5/6KK+5f2Mv2NPh/8AHn9nfSb6702O6uLhb3+29aXUZv7R06/ivbZoNPhtC0UbW9xYrLicLcMss7MTEIdpAPhqpp7eWCTa0cykdmr92JP+CbXwK+L2j2mpeJvh3b/21b6XHl4NQuLKL93GPLt/Lt5I45PLjAj8z2r8x/8Agr38Cf8AhS/7butWen6fb6b4f1LTNPu9EihP7n7IlslsBH/sJJbyxj/rlQB6R+zB8Qv2Q4fDulQ64uteF9ct7W3N5d+JPDI1y1ku8fvHjkt5fM8veOn2evqzwp+3Z+y/4LtY7i6+LugXUenx/uLXQvB+s/aZP+mccdxbxxR/9/a/GWigD94vgf8A8Fmf2PfHdprFr4m8afFnwNFY28H2KTVvC8dx9vkk8zzPLjs5Ljy/L/d/6yT95XyL/wAFPv2/fg/8YPj7pN98L9duNa8O2GgxWUl3deH5bOSScXNzIfkZVOPLki5xivzXooA//9k=

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

        parsed = urlparse.urlparse(self.path)
        base64_img = parse_qs(parsed.query)['def'][0]
        print(base64_img)

        # image1 = Image.frombytes('RGB', (28, 28), decodebytes(bytes(base64_img)))
        # image1.save("4.jpeg")

        # img_name = str(token_urlsafe(4))
        img_name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        im = open(img_name + '.jpeg', "w")
        im.write(decodebytes(bytes(base64_img)))
        im.close()

        img_path = img_name+'.jpeg'
        img = image.load_img(img_path)
        image_x = 28
        image_y = 28
        img = img.resize((image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, image_x, image_y))
        img = img / 255.0
        pred_probab = model.predict(img)[0]
        pred_class = list(pred_probab).index(max(pred_probab))

        self.wfile.write(str(class_names[pred_class] + " " + str(int(round(100*max(pred_probab)))) + "%").encode())


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
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

    print("Model loaded")

    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    print("Server started")
    httpd.serve_forever()
