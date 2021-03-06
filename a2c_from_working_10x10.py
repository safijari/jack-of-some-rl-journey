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
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.001):
        super(SnakeModel, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.model, _, _ = make_main_model(input_shape, num_actions)
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
            tf.keras.layers.Dense(num_actions)
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'),
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
        neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(actions, 4))
        _policy_loss = neglogpac*tf.squeeze(advs)
        policy_loss = tf.reduce_mean(_policy_loss)

        vpred = model.vcall(states)
        value_loss = tf.reduce_mean(tf.square(vpred-rewards))

        entropy = tf.reduce_mean(calc_entropy(logits))

        loss = policy_loss + value_loss * 0.5  - 0.05*entropy

    var_list = tape.watched_variables()
    grads = tape.gradient(loss, var_list)
    grads, _ = tf.clip_by_global_norm(grads, 0.5)
    model.optimizer.apply_gradients(zip(grads, var_list))
    return loss, policy_loss, value_loss, entropy

def main():
    wandb.init('snake-a2c')
    gs = 20
    main_gs = 22
    batch_size = 16
    num_actions = 4
    num_envs = 64
    num_fruits = 1
    envs = [gym.make('snakenv-v0', gs=gs, main_gs=main_gs, num_fruits=num_fruits) for _ in range(num_envs)]

    test_env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs, num_fruits=num_fruits)
    test_env.reset()
    test_env.render()

    state = np.stack([env.reset() for env in envs])

    model = SnakeModel((80*2 + 16, 80*2 + 16, 1), num_actions)
    model.model.load_weights('./10x10_fully_successful_sprites/10838016_main.h5')

    sarsdv = []
    pbar = tqdm()

    def _a(l, idx):
        return np.concatenate([m[idx] for m in l])

    rew = np.zeros((num_envs, 1))

    steps = 0
    num_eps = 0
    steps_since_last_test = 0
    while True:
        sarsdv = []
        for _ in range(batch_size):
            for env in envs[:4]:
                env.render()
            steps += num_envs
            steps_since_last_test += num_envs
            pbar.update(num_envs)
            action_logits, action_probs, value = model(state)

            action = [np.random.choice(range(num_actions), p=ap.numpy()) for ap in action_probs]
            # next_s, reward, done, info_dict = env.step(action)
            next_s, reward, done, info_dict = zip(*[env.step(a) for env, a in zip(envs, action)])
            next_s = np.stack(next_s)
            reward = np.expand_dims(np.stack(reward), -1)
            done = np.expand_dims(np.stack(done), -1)

            rew += reward

            sarsdv.append((state, action, reward, next_s, done, value))

            state = next_s.copy()

            for i, env in enumerate(envs):
                if done[i]:
                    num_eps += 1
                    state[i] = env.reset()
                    wandb.log({'episode_reward': rew[i], 'num_eps': num_eps, 'score': info_dict[i]['score']}, step=steps)
                    rew[i] = 0

        if steps_since_last_test >= 250000:
            folder = '20x20aaaaaaaa'
            if not os.path.exists(folder):
                os.mkdir(folder)
            model.model.save(f'{folder}/{steps}_main.h5')
            model.policy_head.save(f'{folder}/{steps}_policy.h5')
            model.value_head.save(f'{folder}/{steps}_value.h5')
            print('testing')
            steps_since_last_test = 0
            trew = 0
            tstate = test_env.reset()
            tdone = False
            while not tdone:
                test_env.render()
                action_logits, action_probs, value = model(_e(tstate))
                action = np.argmax(action_probs)
                tstate, step_rew, tdone, info_dict = test_env.step(action)
                trew += step_rew

            wandb.log({'test_reward': trew, 'test_score': info_dict['score']}, step=steps)

        _, _, R = model(state)

        discounted_rewards = []
        for _, _, r, _, d, _ in reversed(sarsdv):
            R = r + model.gamma * R * (1-d)
            discounted_rewards.append(R)

        discounted_rewards = np.concatenate(np.array(list(reversed(discounted_rewards))))

        states = _a(sarsdv, 0)
        values = _a(sarsdv, -1)
        actions = _a(sarsdv, 1)


        loss = train(model, states.astype('float32'), discounted_rewards.astype('float32'), values.astype('float32'), actions.astype('int32'))
        wandb.log(dict(zip(['loss', 'policy_loss', 'value_loss', 'entropy'], loss)), step=steps)


if __name__ == '__main__':
    main()
