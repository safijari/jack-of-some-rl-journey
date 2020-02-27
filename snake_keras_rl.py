import argh
import traceback
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

def make_model(shape, num_actions):
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=shape))
    model.add(Convolution2D(32, (1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))
    if shape[0] > 10:
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(MaxPooling2D())
        model.add(Activation('relu'))
    if shape[0] > 20:
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(MaxPooling2D())
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(MaxPooling2D())
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(MaxPooling2D())
        model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
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


def main(grid_shape=5, shape=4, winsize=2, test=False, num_max_test=200, visualize_training=False, start_steps=0, randseed=None, human_mode_sleep=0.02):
    INPUT_SHAPE = (grid_shape, grid_shape)
    WINDOW_LENGTH = winsize
    input_shape = INPUT_SHAPE + (WINDOW_LENGTH,)

    class SnakeProcessor(Processor):
        def process_observation(self, observation):
            assert observation.shape == (grid_shape, grid_shape, 1)
            return observation.astype('uint8')  # saves storage in experience memory

        def process_state_batch(self, batch):
            # We could perform this processing step in `process_observation`. In this case, however,
            # we would need to store a `float32` array instead, which is 4x more memory intensive than
            # an `uint8` array. This matters if we store 1M observations.
            batches = []
            if len(batch.shape) > 4:
                for i in range(batch.shape[0]):
                    batches.append(np.dstack([b for b in batch[0, :]]))
            batch = np.stack(batches, 0)
            processed_batch = batch.astype('float32') / 255.
            return processed_batch

        def process_reward(self, reward):
            return reward

    try:
        randseed = int(randseed)
        print(f"set seed to {randseed}")
    except Exception:
        print(f"failed to intify seed of {randseed}, making it None")
        randseed = None

    env = gym.make('snakenv-v0', gs=shape, seed=randseed, human_mode_sleep=human_mode_sleep, rand_grid_loc_always=(not test), full_grid_size=grid_shape)
    np.random.seed(123)
    env.seed(123)

    # input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = make_model(input_shape, 4)

    memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
    processor = SnakeProcessor()

    start_policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=0, value_min=0,
        value_test=0, nb_steps=500000)
    policy = BoltzmannQPolicy(tau=0.25)

    interval = 5000

    dqn = DQNAgent(model=model,
                   nb_actions=4,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=2000,
                   gamma=.99,
                   target_model_update=interval,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(), metrics=['mae'])
    weights_filename = 'dqn_snake_weights.h5f'

    if not test:
        if os.path.exists('starting_weights.h5'):
            print('loadin!')
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
                model.load_weights(weights_filename)
            except Exception:
                print("weights not found, waiting")
                traceback.print_exc()
                time.sleep(10)
                continue
            dqn.test(env, nb_episodes=10, visualize=visualize_training, nb_max_episode_steps=num_max_test)
            time.sleep(3)

if __name__ == '__main__':
    argh.dispatch_command(main)
