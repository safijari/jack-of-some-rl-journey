import tensorflow as tf
from tqdm import tqdm
import time
from collections import deque
import numpy as np
from tf2_common import make_main_model
from dqn_tf2 import RunCfg, run_full_episode, Exp, Episode

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

import random
import gym
import wandb

class SnakeModel(Model):
    def __init__(self, input_shape, num_actions):
        super(SnakeModel, self).__init__()
        # image_shape should be (h, w, channels)
        self.num_actions = num_actions
        self.image_shape = input_shape
        self.model = make_main_model(input_shape, num_actions, include_finals=False)
        self.logits_dense = tf.keras.layers.Dense(512)
        self.value_dense = tf.keras.layers.Dense(512)
        self.logits = tf.keras.layers.Dense(num_actions, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    @tf.function
    def get_policy(self, x):
        x = x / 255.0
        latent = self.logits_dense(self.model(x))
        return self.logits((latent))

    @tf.function
    def call(self, x):
        latent = self.model(x/255.0)
        return self.logits(self.logits_dense(latent)), self.value(self.value_dense(latent))



class Agent:
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.999

        self.action_size = 3
        self.a2c = SnakeModel((128, 128, 1), self.action_size)
        self.opt = optimizers.Adam(lr=self.lr, )

        self.entropy_coefficient = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        # lr=7e-4
        # alpha=0.99
        # epsilon=1e-5
        # self.opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=alpha, epsilon=epsilon)


        self.rollout = 128
        self.batch_size = 128

    def get_action(self, state, stochastic=False):
        state = tf.constant(np.array([state], dtype='float32'))
        policy = self.a2c.get_policy(state)
        policy = np.array(policy)[0]
        if stochastic:
            action = np.random.choice(self.action_size, p=policy)
        else:
            action = np.argmax(policy)
        return action

    def update(self, state, next_state, reward, done, action):
        sample_range = np.arange(len(state)-1)
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.batch_size]

        state = np.stack(([state[i] for i in sample_idx])).astype('float32')
        next_state = np.stack(([next_state[i] for i in sample_idx])).astype('float32')
        reward = np.stack(([reward[i] for i in sample_idx])).astype('float32')
        done = np.stack(([done[i] for i in sample_idx])).astype('float32')
        action = np.stack(([action[i] for i in sample_idx])).astype('int32')
        return self._update(state, next_state, reward, done, action)

    @tf.function
    def _update(self, state, next_state, reward, done, action):
        # a2c_variable = self.a2c.trainable_variables
        with tf.GradientTape() as tape:
            policy, current_value = self.a2c(state)
            _, next_value = self.a2c(next_state)
            current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
            target = reward
            value_loss = tf.reduce_mean(tf.square(target - current_value) * self.vf_coef)

            entropy = tf.reduce_mean(- policy * tf.math.log(policy+1e-8)) * self.entropy_coefficient
            action = (action)
            onehot_action = tf.one_hot(action, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
            adv = tf.stop_gradient(target - current_value)
            pi_loss = -tf.reduce_mean(tf.math.log(action_policy+1e-8) * adv) - entropy

            total_loss = pi_loss + value_loss

        grads = tape.gradient(total_loss, self.a2c.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.opt.apply_gradients(zip(grads, self.a2c.trainable_variables))
        return total_loss, pi_loss, value_loss, entropy

    def run(self):
        gs = 6
        main_gs = gs
        max_possible_reward = gs**2 - 2
        cfg = RunCfg(total_steps=1000000,
                     batch_size=64,
                     gs=gs,
                     main_gs=main_gs,
                     max_possible_reward=max_possible_reward,
                     target_model_steps=20000,
                     test_steps=5000,
                     stacking=1,
                     steps_between_train=4,
                     starting_temperature=3,
                     temperature_decay_idx=2000000)

        env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs)
        episodes = 0

        i = 0

        pbar = tqdm()

        while True:
            num_obs = 0
            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []

            while num_obs < self.rollout:
                ep = run_full_episode(env, self)
                episodes += 1
                num_obs += len(ep.exps)
                i += len(ep.exps)
                pbar.update(1)
                wandb.log({'episode_reward': ep.total_rew, 'episode_length': len(ep.exps), 'episodes': episodes}, step=i)
                num_obs += len(ep.exps)
                for exp in ep.exps:
                    state_list.append(exp.obs)
                    next_state_list.append(exp.obs_next)
                    reward_list.append(exp.discounted_rew)
                    done_list.append(float(exp.done))
                    action_list.append(int(exp.action))

            loss, pi_loss, val_loss, entropy = self.update(
                state=np.array(state_list, dtype='float32'),
                next_state=np.array(next_state_list, dtype='float32'),
                reward=reward_list, done=done_list, action=action_list)

            wandb.log({'loss': loss, 'pi_loss': pi_loss, 'value_loss': val_loss, 'entroy': entropy}, step=i)

def run_full_episode(env, model: Agent, render_time=0, test=False):
    exps = []
    state = env.reset()
    if render_time:
        env.render()
        time.sleep(render_time)
    done = False

    score_so_far = 0
    while not done:
        action = model.get_action(state, (not test))
        state_old = state
        state, rew, done, _ = env.step(action)
        exps.append(Exp(state_old, state, action, rew, 0, done))
        score_so_far += rew

        if render_time:
            env.render()
            time.sleep(render_time)

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

if __name__ == '__main__':
    wandb.init(project='snake-a2c')
    agent = Agent()
    agent.run()
