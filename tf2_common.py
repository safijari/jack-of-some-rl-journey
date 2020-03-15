import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model

def make_main_model(input_shape, num_actions):
    layers = [
            Conv2D(32, 8, padding='valid', activation='relu', strides=(4, 4), input_shape=input_shape, kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 4, padding='valid', activation='relu', strides=(2, 2), kernel_initializer='random_uniform', bias_initializer='zeros'),
            # Conv2D(64, 4, padding='valid', activation='relu', strides=(2, 2), kernel_initializer='random_uniform', bias_initializer='zeros'),
            # Conv2D(128, 4, padding='valid', activation='relu', strides=(2, 2), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 3, padding='valid', activation='relu', strides=(1, 1), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Flatten(),
        ]

    policy_head = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(num_actions)
    ])
    value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(1)
    ])

    return tf.keras.Sequential(layers), policy_head, value_head

# taken from https://github.com/openai/baselines/blob/tf2/baselines/deepq/deepq_learner.py

def make_eights_model(input_shape, num_actions):
    assert input_shape[0] % 8 == 0, "this model works in cells of 8"
    inp = Input(shape=input_shape)
    layers = [
            Conv2D(64, 8, padding='valid', activation='relu', strides=(8, 8), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(64, 8, padding='same', activation='relu', strides=(4, 4), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Conv2D(256, 10, padding='valid', activation='relu', strides=(1, 1), kernel_initializer='random_uniform', bias_initializer='zeros'),
            Flatten(),
        ]

    x = inp
    for l in layers:
        x = l(x)

    policy_head = [
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(num_actions)
    ]
    value_head = [
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(1)
    ]
    policy = policy_head[-1](policy_head[-2](x))
    value = value_head[-1](value_head[-2](x))

    return Model(inputs=inp, outputs=x), Model(inputs=inp, outputs=policy_head), Model(inputs=inp, outputs=value_head)

if __name__ == '__main__':
    model = make_eights_model((320, 320, 1), 4)

    # print(model.summary())
    print(model(np.zeros((1, 320, 320, 1))))
