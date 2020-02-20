import argh
import gym
import numpy as np
import tensorflow.keras as kr
from rl.agents.dqn import DQNAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from snake_gym import SnakeEnv
from rl.callbacks import WandbLogger

INPUT_SHAPE = (4, 4)
WINDOW_LENGTH = 2

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

def make_model(shape, num_actions):
    model = Sequential()
    model.add(Convolution2D(32, (1, 1), padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (1, 1), padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Flatten())
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


if __name__ == '__main__':
    env = gym.make('snakenv-v0')
    np.random.seed(123)
    env.seed(123)

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = make_model(input_shape, 5)

    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
    processor = SnakeProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
        value_test=0, nb_steps=500000)

    interval = 20000

    dqn = DQNAgent(model=model,
                   nb_actions=5,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=20000,
                   gamma=.99,
                   target_model_update=interval,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    if False:
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'dqn_{}_weights.h5f'.format('snake')
        checkpoint_weights_filename = 'dqn_' + 'snake' + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format('snake')
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=interval)]
        callbacks += [FileLogger(log_filename, interval=500)]
        dqn.fit(env, callbacks=callbacks, nb_steps=5000000, log_interval=10000, visualize=False)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=100)
    else:
        weights_filename = 'dqn_snake_weights.h5f'
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=100, visualize=True, nb_max_episode_steps=200)

