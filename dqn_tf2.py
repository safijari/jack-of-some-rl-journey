import tensorflow as tf
import os
import cv2
from tqdm import tqdm
from typing import List
import random
import numpy as np
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import gym
from snake_gym import SnakeEnv
import wandb

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

@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

class SnakeModel(Model):
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.001):
        super(SnakeModel, self).__init__()
        # image_shape should be (h, w, channels)
        self.num_actions = num_actions
        self.image_shape = input_shape
        self.gamma = gamma
        self.lr = lr
        with tf.name_scope('model'):
            self.model = make_main_model(input_shape, num_actions)
        with tf.name_scope('target_model'):
            self.target_model = make_main_model(input_shape, num_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.eps = tf.Variable(0., name='eps')

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    def call(self, x):
        return self.model(x)

    @tf.function
    def predict(self, obs: tf.Tensor, stochastic=True, override_eps=0, update_eps=-1):
        obs = obs/255
        if len(obs.shape) == 3:
            obs = tf.expand_dims(obs, 0)
        # note: obs is obvs a batch
        q_vals = self.model(obs)
        deterministic_actions = tf.argmax(q_vals, axis=1)
        if not stochastic:
            return deterministic_actions

        batch_size = tf.shape(obs)[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
        # eps = self.eps if override_eps <= 0 else override_eps
        eps = override_eps
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
        if update_eps >= 0:
            self.eps.assign(update_eps)

        return stochastic_actions

    @tf.function
    def train(self, obs0, actions, rewards, obs1, dones):  #, importance_weights):
        obs0 = obs0 / 255
        obs1 = obs1 / 255
        with tf.GradientTape() as tape:
            q_t = self.model(obs0)
            # the one hot multiplier simply sets all non actioned
            # q value locations to 0, and then the reduce sum gets
            # rid of them
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(
                actions, self.num_actions, dtype=tf.float32), 1)
            q_tp1 = self.target_model(obs1)

            # if self.double_q:
            #     q_tp1_using_online_net = self.q_network(obs1)
            #     q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            #     q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)

            q_tp1_best = tf.reduce_max(q_tp1, 1)  # picks the best Q value for each batch, along axis 1

            dones = tf.cast(dones, q_tp1_best.dtype)

            q_tp1_best_masked = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best_masked

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

            errors = huber_loss(td_error)

            # weighted_error = tf.reduce_mean(importance_weights * errors)

        grads = tape.gradient(errors, self.model.trainable_variables)

        # clipping?
        grads_and_vars = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return errors, td_error

    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.model.trainable_variables
        target_q_vars = self.target_model.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)


from dataclasses import dataclass


@dataclass
class Exp:
    obs: int
    obs_next: int
    action: int
    rew: int
    discounted_rew: int
    done: bool

    def __repr__(self):
        return f"(Exp) obs, obs_next, action: {self.action}, rew: {self.rew}, disrew: {self.discounted_rew}, done: {self.done}"

@dataclass
class Episode:
    exps: list
    total_rew: int

    def get_exp(self, idx, shift: int = 0) -> Exp:
        idx = max(0, idx + shift)
        return self.exps[idx + shift]

    def get_stacked_exp(self, idx, stacking=1):
        obs = []
        obs_next = []
        main_exp = self.get_exp(idx)
        for s in range(stacking):
            exp = self.get_exp(idx, -s)
            obs.append(exp.obs)
            obs_next.append(exp.obs_next)

        return Exp(np.dstack(obs), np.dstack(obs_next),
                   main_exp.action, main_exp.rew, main_exp.discounted_rew, main_exp.done
                   )

    def __repr__(self):
        l = [f"(Episode) Total reward: {self.total_rew}"] + [str(e) for e in self.exps]
        return '\n'.join(l)

def run_full_episode(env, model: SnakeModel, override_eps=0, render_time=0, test=False):
    exps = []
    state = env.reset()
    if render_time:
        env.render()
        time.sleep(render_time)
    done = False

    while not done:
        action = int(model.predict(state, stochastic=(not test), override_eps=override_eps))
        state_old = state
        state, rew, done, _ = env.step(action)
        exps.append(Exp(state_old, state, action, rew, 0, done))

        if render_time:
            env.render()
            time.sleep(render_time)

    last_rew = 0
    for e in reversed(exps):
        if e.done:
            e.discounted_rew = e.rew
        else:
            e.discounted_rew = e.rew + model.gamma * last_rew
        last_rew = e.discounted_rew

    return Episode(exps, sum(e.rew for e in exps))

@dataclass
class EpFrameIdx:
    ep: Episode
    idx: int

    def get_stacked_exp(self, stacking=1):
        return self.ep.get_stacked_exp(self.idx, stacking)

from collections import deque

class EpisodicReplayBuffer:
    def __init__(self, max_steps=100000):
        self.max_steps = max_steps
        self.frame_indices: List[EpFrameIdx] = []
        self.episode_reward_counter = deque(maxlen=100)

    def __len__(self):
        return len(self.frame_indices)

    def add_new_episode(self, ep: Episode):
        for i in range(len(ep.exps)):
            self.frame_indices.append(EpFrameIdx(ep, i))

        self.episode_reward_counter.append(ep.total_rew)

        self.frame_indices = self.frame_indices[-self.max_steps:]

    def sample_frames(self, batch_size, stacking=1):
        epfridx = random.sample(self.frame_indices, batch_size)

        return [e.get_stacked_exp(stacking) for e in epfridx]

def experience_samples_to_training_input(samples):
    obs = []
    obs_next = []
    actions = []
    rewards = []
    dones = []
    for s in samples:
        obs.append(s.obs)
        obs_next.append(s.obs_next)
        actions.append(s.action)
        rewards.append(s.rew)
        dones.append(s.done)

    return tf.convert_to_tensor(np.stack(obs, 0)), tf.convert_to_tensor(actions, dtype='int32'), tf.convert_to_tensor(rewards, dtype='float32'), tf.convert_to_tensor(np.stack(obs_next, 0)), tf.convert_to_tensor(dones, dtype='bool')

def get_episodes(env, model, num_episodes, eps=0.5):
    episodes = [run_full_episode(env, model, override_eps=eps)
                for i in range(int(num_episodes*3))]
    episodes = sorted(episodes, key=lambda e: e.total_rew)[-num_episodes:]
    return episodes

@dataclass
class RunCfg:
    total_steps: int
    batch_size: int
    gs: int
    main_gs: int
    max_possible_reward: int
    target_model_steps: int
    test_steps: int
    ending_eps: int
    stacking: int

def main():
    last_test_rewards = deque(maxlen=10)
    gs = 6
    main_gs = gs
    max_possible_reward = gs**2 - 2
    eps = 0.8,
    cfg = RunCfg(total_steps=1000000,
                 batch_size=64,
                 gs=gs,
                 main_gs=main_gs,
                 max_possible_reward=max_possible_reward,
                 target_model_steps=1000,
                 test_steps=100,
                 ending_eps=0.05,
                 stacking=1)

    wandb.init(project='tf2-messing-around',
               config=vars(cfg))
    model = SnakeModel((128, 128, cfg.stacking), 3)
    print(model.summary())
    env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs)

    replay = EpisodicReplayBuffer(100000)

    while len(replay) < 5000:
        for ep in get_episodes(env, model, 10):
            replay.add_new_episode(ep)

    for i in tqdm(range(cfg.total_steps)):
        for ep in get_episodes(env, model, 2, eps=eps):
            replay.add_new_episode(ep)

        sample = replay.sample_frames(cfg.batch_size, stacking=cfg.stacking)
        inp = experience_samples_to_training_input(sample)
        l = model.train(*inp)

        loss = float(tf.reduce_mean(l[0]))

        if i and i % cfg.target_model_steps == 0:
            model.update_target()

        if i % cfg.test_steps == 0:
            render_time = 0.05 if os.path.exists('/tmp/vis') else 0
            rew = np.mean([run_full_episode(env, model, test=True, render_time=render_time).total_rew for _ in range(5)])
            last_test_rewards.append(rew)
            wandb.log({'test_reward': rew}, step=i)

        if len(last_test_rewards) >= 10:
            avg_test_rewards = float(np.mean(last_test_rewards))
            # eps = max(0, 1 - avg_test_rewards/cfg.max_possible_reward)
            eps = 0.1

        wandb.log({'loss': loss, 'average_episode_reward': np.mean(replay.episode_reward_counter),
                   'eps': eps}, step=i)


if __name__ == '__main__':
    main()
