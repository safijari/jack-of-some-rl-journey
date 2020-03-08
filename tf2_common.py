import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def make_main_model(input_shape, num_actions, include_finals=True):
    layers = [
            Conv2D(32, 8, padding='valid', activation='relu', strides=(4, 4), input_shape=input_shape, kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 4, padding='valid', activation='relu', strides=(2, 2), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 4, padding='valid', activation='relu', strides=(2, 2), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 3, padding='valid', activation='relu', strides=(1, 1), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Flatten(),
        ]
    if include_finals:
        d1 = Dense(512, activation='relu')
        d = Dense(num_actions)
        layers.append(d1)
        layers.append(d)
    return tf.keras.Sequential(layers)

# taken from https://github.com/openai/baselines/blob/tf2/baselines/deepq/deepq_learner.py

def make_eights_model(input_shape, num_actions, include_finals=True):
    assert input_shape[0] % 8 == 0, "this model works in cells of 8"
    layers = [
            Conv2D(64, 8, padding='valid', activation='relu', strides=(8, 8), input_shape=input_shape, kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 8, padding='valid', activation='relu', strides=(4, 4), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(256, 10, padding='valid', activation='relu', strides=(1, 1), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Flatten(),
        ]
    if include_finals:
        d1 = Dense(512, activation='relu')
        d = Dense(num_actions)
        layers.append(d1)
        layers.append(d)
    return tf.keras.Sequential(layers)
