import argh
import os
import time
import gym
import numpy as np
import tensorflow.keras as kr
from rl.agents.dqn import DQNAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, MaxPooling2D
from keras.optimizers import Adam

from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy, BoltzmannGumbelQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from snake_gym import SnakeEnv
from rl.callbacks import WandbLogger
# def make_model(shape, num_actions):
#     model = Sequential()
#     model.add(Permute((2, 3, 1), input_shape=shape))
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(128, (3, 3), padding='same'))
#     model.add(MaxPooling2D())
#     model.add(Activation('relu'))
#     model.add(Convolution2D(256, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(256, (3, 3), padding='same'))
#     model.add(MaxPooling2D())
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(num_actions))
#     model.add(Activation('linear'))
#     print(model.summary())
#     return model

def make_ridiculous_model(shape, num_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=shape))
    model.add(Convolution2D(512, (40, 40), padding='valid'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

def make_model(shape, num_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=shape))
    model.add(Convolution2D(32, (8, 8), padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (8, 8), padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (4, 4), padding='valid'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    # model.add(Convolution2D(128, (3, 3), padding='valid'))
    # model.add(MaxPooling2D())
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, (3, 3), padding='valid'))
    # model.add(MaxPooling2D())
    # model.add(Activation('relu'))
    # model.add(Convolution2D(256, (3, 3), padding='valid'))
    # model.add(MaxPooling2D())
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

# def make_model(shape, num_actions):
#     model = Sequential()
#     model.add(Permute((2, 3, 1), input_shape=shape))
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(128, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(num_actions))
#     model.add(Activation('linear'))
#     print(model.summary())
#     return model


def main(main_grid_shape=40, shape=4, winsize=2, test=False, num_max_test=200):
    INPUT_SHAPE = (main_grid_shape, main_grid_shape)
    WINDOW_LENGTH = winsize

    class SnakeProcessor(Processor):
        def process_observation(self, observation):
            # assert observation.ndim == 1, str(observation.shape)  # (height, width, channel)
            assert observation.shape == INPUT_SHAPE
            return observation.astype('uint8')  # saves storage in experience memory

        def process_state_batch(self, batch):
            # We could perform this processing step in `process_observation`. In this case, however,
            # we would need to store a `float32` array instead, which is 4x more memory intensive than
            # an `uint8` array. This matters if we store 1M observations.
            processed_batch = batch.astype('float32') / 255.
            return processed_batch

        def process_reward(self, reward):
            return reward

    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
    env = gym.make('snakenv-v0', gs=shape, main_gs=main_grid_shape)
    np.random.seed(123)
    env.seed(123)

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = make_model(input_shape, 3)

    processor = SnakeProcessor()

    policy = BoltzmannQPolicy(tau=0.25)

    interval = 10000

    dqn = DQNAgent(model=model,
                   nb_actions=3,
                   enable_double_dqn=True,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=2000,
                   gamma=.99,
                   target_model_update=interval,
                   train_interval=500,
                   delta_clip=1.)

    dqn.compile(Adam(), metrics=['mae'])
    weights_filename = 'dqn_snake_weights.h5f'

    if not test:
        if os.path.exists('starting_weights.h5'):
            model.load_weights('starting_weights.h5')
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'dqn_{}_weights.h5f'.format('snake')
        checkpoint_weights_filename = 'dqn_' + 'snake' + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format('snake')
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=interval)]
        callbacks += [ModelIntervalCheckpoint(weights_filename, interval=interval)]
        callbacks += [FileLogger(log_filename, interval=500)]
        callbacks += [WandbLogger(project="snake-rl")]
        dqn.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=10000, visualize=visualize_training, nb_max_start_steps=start_steps)

        # After training is done, we save the final weights one more time.
        # dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        # dqn.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=100)
    else:
        while True:
            try:
                dqn.load_weights(weights_filename)
            except Exception:
                print("weights not found, waiting")
            dqn.test(env, nb_episodes=10, visualize=visualize_training, nb_max_episode_steps=num_max_test)
            time.sleep(3)

if __name__ == '__main__':
    argh.dispatch_command(main)
