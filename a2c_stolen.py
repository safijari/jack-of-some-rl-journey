import tensorflow as tf
from tqdm import tqdm
import os
import cv2
from tqdm import tqdm
from typing import List
import random
import numpy as np
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax
from tensorflow.keras import Model
import gym
from snake_gym import SnakeEnv
import wandb
from tf2_common import make_main_model

@tf.function
def calc_entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

class SnakeModel(Model):
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.0001):
        super(SnakeModel, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.model = make_main_model(input_shape, num_actions, False)
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_actions)
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.num_actions = num_actions
        self.image_shape = input_shape
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.act = tf.keras.layers.Activation('softmax')

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    @tf.function
    def call(self, x):
        latent = self.model(x/255)
        p = self.policy_head(latent)
        return p, self.act(p), self.value_head(latent)

    @tf.function
    def pcall(self, x):
        latent = self.model(x/255)
        return self.policy_head(latent)

    @tf.function
    def vcall(self, x):
        latent = self.model(x/255)
        return self.value_head(latent)

def _e(s):
    return np.expand_dims(s, 0).astype('float32')

@tf.function
def train(model, states, rewards, values, actions):

    advs = rewards - values
    with tf.GradientTape() as tape:
        logits = model.pcall(states)
        neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(actions, 3))
        _policy_loss = neglogpac*tf.squeeze(advs)
        policy_loss = tf.reduce_mean(_policy_loss)

        vpred = model.vcall(states)
        value_loss = tf.reduce_mean(tf.square(vpred-rewards))

        entropy = tf.reduce_mean(calc_entropy(logits))

        loss = policy_loss + value_loss * 0.5  - 0.001*entropy

    var_list = tape.watched_variables()
    grads = tape.gradient(loss, var_list)
    model.optimizer.apply_gradients(zip(grads, var_list))
    return loss, policy_loss, value_loss, entropy

def main():
    wandb.init('snake-a2c')
    gs = 8
    main_gs = 8
    batch_size = 512
    num_actions = 3
    env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs)

    state = env.reset()

    model = SnakeModel((84, 84, 1), num_actions)

    sarsdv = []
    pbar = tqdm()

    def _a(l, idx):
        return np.array([m[idx] for m in l])

    rew = 0
    episode_rewards = []

    steps = 0
    num_eps = 0
    while True:
        sarsdv = []
        for _ in range(batch_size):
            steps += 1
            pbar.update(1)
            action_logits, action_probs, value = model(_e(state))
            action = np.random.choice(range(num_actions), p=action_probs[0].numpy())
            next_s, reward, done, _ = env.step(action)

            rew += reward

            sarsdv.append((state, action, reward, next_s, done, value))

            state = next_s

            if done:
                num_eps += 1
                state = env.reset()
                wandb.log({'episode_reward': rew, 'num_eps': num_eps}, step=steps)
                rew = 0

        R = 0

        if not done:
            _, _, R = model(_e(state))
            R = float(R.numpy()[0])

        discounted_rewards = []
        for _, _, r, _, d, _ in reversed(sarsdv):
            if d:
                R = 0
            R = r + model.gamma * R
            discounted_rewards.append(R)


        discounted_rewards = np.expand_dims(np.array(list(reversed(discounted_rewards))), 1)

        states = _a(sarsdv, 0)
        values = _a(sarsdv, -1)[:, :, 0]
        actions = _a(sarsdv, 1)

        loss = train(model, states.astype('float32'), discounted_rewards.astype('float32'), values.astype('float32'), actions.astype('int32'))
        wandb.log(dict(zip(['loss', 'policy_loss', 'value_loss', 'entropy'], loss)), step=steps)


if __name__ == '__main__':
    main()
