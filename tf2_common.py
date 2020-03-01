import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def make_main_model(input_shape, num_actions):
    return tf.keras.Sequential(
        [
            Conv2D(32, 8, padding='valid', activation='relu', strides=(4, 4), input_shape=input_shape),
            Conv2D(64, 4, padding='valid', activation='relu', strides=(2, 2)),
            Conv2D(64, 3, padding='valid', activation='relu', strides=(1, 1)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(num_actions),
        ])

# taken from https://github.com/openai/baselines/blob/tf2/baselines/deepq/deepq_learner.py
