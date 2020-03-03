import tensorflow as tf
import time
from collections import deque
import numpy as np
from tf2_common import make_main_model
from dqn_tf2 import RunCfg

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model

import random
import gym

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
    def call(self, x):
        latent = self.model(x)
        return self.logits(self.logits_dense(latent)), self.value(self.value_dense(latent))

class Agent:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.99

        self.action_size = 3
        self.a2c = SnakeModel((128, 128, 1), self.action_size)
        self.opt = optimizers.Adam(lr=self.lr, )

        self.rollout = 128
        self.batch_size = 128
        self.state_size = 4

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.a2c(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    @tf.function
    def update(self, state, next_state, reward, done, action):
        sample_range = np.arange(self.rollout)
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.batch_size]

        state = [state[i] for i in sample_idx]
        next_state = [next_state[i] for i in sample_idx]
        reward = [reward[i] for i in sample_idx]
        done = [done[i] for i in sample_idx]
        action = [action[i] for i in sample_idx]

        a2c_variable = self.a2c.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(a2c_variable)
            _, current_value = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
            _, next_value = self.a2c(tf.convert_to_tensor(next_state, dtype=tf.float32))
            current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
            target = tf.stop_gradient(self.gamma * (1-tf.convert_to_tensor(done, dtype=tf.float32)) * next_value + tf.convert_to_tensor(reward, dtype=tf.float32))
            value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

            policy, _  = self.a2c(tf.convert_to_tensor(state, dtype=tf.float32))
            entropy = tf.reduce_mean(- policy * tf.math.log(policy+1e-8)) * 0.1
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            onehot_action = tf.one_hot(action, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
            adv = tf.stop_gradient(target - current_value)
            pi_loss = -tf.reduce_mean(tf.math.log(action_policy+1e-8) * adv) - entropy

            total_loss = pi_loss + value_loss

        grads = tape.gradient(total_loss, a2c_variable)
        self.opt.apply_gradients(zip(grads, a2c_variable))

    def run(self):
        # last_test_rewards = deque(maxlen=10)
        gs = 4
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

        # num_envs = 4

        env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs)
        state = env.reset()

        episode = 0
        score = 0

        while True:
            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []

            for _ in range(self.rollout):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                env.render()
                time.sleep(0.005)

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)

                state = next_state

                if done:
                    print(episode, score)
                    state = env.reset()
                    episode += 1
                    score = 0

            self.update(
                state=state_list, next_state=next_state_list,
                reward=reward_list, done=done_list, action=action_list)


if __name__ == '__main__':
    agent = Agent()
    agent.run()
