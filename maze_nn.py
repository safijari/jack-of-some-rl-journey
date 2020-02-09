import tensorflow.keras as kr
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

def preprocess_image(im, image_size=64, expand=True):

    im = (cv2.resize(im, (image_size, image_size)) - 128.0)/128

    if expand:
        return np.expand_dims(im, 0)
    return im

def predict_on_model(im, m, return_raw):
    o = m.predict(preprocess_image(im))
    if return_raw:
        return o[0]
    return np.argmax(o[0])

def create_maze_solving_network(image_size=64, num_actions=4):
    model = kr.models.Sequential()
    model.add(Conv2D(16, 8, input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 4, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile('sgd', 'mse')
    return model

if __name__ == '__main__':
    model = create_maze_solving_network()
    # print(model.summary())
    im = cv2.imread('/home/jack/test.jpg')

    print(predict_on_model(im, model))
