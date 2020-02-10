import tensorflow.keras as kr
import tensorflow.keras.backend as K
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# Ref https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368

def preprocess_image(im, image_size=64, expand=True):
    im = cv2.resize(im, (image_size, image_size))/255.0

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
    model.add(Conv2D(16, 3, input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    # model = kr.models.Sequential()
    # model.add(Conv2D(64, 8, strides=(4, 4), activation='relu', input_shape=(64, 64, 3)))
    # model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
    # model.add(Conv2D(64, 3, strides=(2, 2), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(num_actions))
    # model.compile(loss='mse', optimizer='adam')

    return model

def masked_mse(args):
    y_true, y_pred, mask = args
    loss = (y_true - y_pred)**2
    loss *= mask
    return K.sum(loss, axis=-1)

def add_rl_loss_to_network(model):
    num_actions = model.output.shape[1]
    y_pred = model.output
    y_true = Input(name='y_true', shape=(num_actions,))
    mask = Input(name='mask', shape=(num_actions,))
    loss_out = Lambda(masked_mse, output_shape=(1,), name='loss')([y_true, y_pred, mask])
    trainable_model = Model(inputs=[model.input, y_true, mask],
                            outputs=loss_out)
    trainable_model.compile(optimizer=Adam(), loss=lambda yt, yp: yp)
    return trainable_model

def transfer_weights_partially(source, target, lr=0.5):
    wts = source.get_weights()
    twts = target.get_weights()

    for i in range(len(wts)):
        twts[i] = lr * wts[i] + (1-lr) * twts[i]
    target.set_weights(twts)

if __name__ == '__main__':
    model = create_maze_solving_network()
    # print(model.summary())
    # im = cv2.imread('/home/jack/test.jpg')

    # print(predict_on_model(im, model))
    add_rl_loss_to_network(model)
