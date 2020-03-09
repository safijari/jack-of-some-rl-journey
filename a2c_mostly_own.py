import tensorflow as tf
import argh
import os
import traceback
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
from tf2_common import make_main_model, make_eights_model

def _a(l, idx):
    return np.concatenate([m[idx] for m in l])

@tf.function
def calc_entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

class SnakeModel(Model):
    def __init__(self, input_shape, num_actions, gamma=0.999, lr=0.0001):
        super(SnakeModel, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.model, self.policy_head, self.value_head = make_main_model(input_shape, num_actions)
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
        v = self.value_head(latent)
        return p, self.act(p), v

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

        loss = policy_loss + value_loss * 0.5  - 0.01*entropy

    var_list = tape.watched_variables()
    grads = tape.gradient(loss, var_list)
    grads, _ = tf.clip_by_global_norm(grads, 0.5)
    model.optimizer.apply_gradients(zip(grads, var_list))
    return loss, policy_loss, value_loss, entropy

def run_train_step(state, rew, pbar, envs, model, num_actions, batch_size, steps, num_envs, num_eps, viz=False):
    sarsdv = []
    num_steps = 0
    for _ in range(batch_size):
        num_steps += num_envs
        pbar.update(num_envs)
        action_logits, action_probs, value = model(state)

        action = [np.random.choice(range(num_actions), p=ap.numpy()) for ap in action_probs]
        next_s, reward, done, info_dict = zip(*[env.step(a) for env, a in zip(envs, action)])
        if viz:
            for i, env in enumerate(envs[:4]):
                env.render()
                if i > 3:
                    continue
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
        return state, num_steps, num_eps

def run_test_step(model, test_env, human_delay=0):
    trew = 0
    tstate = test_env.reset()
    tdone = False
    while not tdone:
        test_env.render()
        time.sleep(human_delay)
        action_logits, action_probs, value = model(_e(tstate))
        action = np.argmax(action_probs)
        tstate, step_rew, tdone, info_dict = test_env.step(action)
        trew += step_rew

    return trew, info_dict['score']

def main(run_name, gs=0, weights_to_load=None, test_only=False, viz_training=False, num_fruits=1):
    assert gs > 3, "grid size must be at least 4"
    assert not os.path.exists(run_name), f"folder for run {run_name} already exists"
    if not test_only:
        wandb.init('snake-a2c', name=run_name)
    summaries_done = False
    main_gs = 40
    batch_size = 50*8
    num_actions = 4
    num_envs = 16
    envs = [gym.make('snakenv-v0', gs=gs, main_gs=main_gs, num_fruits=num_fruits) for _ in range(num_envs)]

    test_env = gym.make('snakenv-v0', gs=gs, main_gs=main_gs, num_fruits=num_fruits)
    test_env.render()

    state = np.stack([env.reset() for env in envs])

    sh = test_env.reset().shape[1]

    model = SnakeModel((sh, sh, 1), num_actions)
    print(model.model.summary())

    if weights_to_load is not None:
        print(f"loading weights from template {weights_to_load}")
        template = weights_to_load
        m = model(state)
        model.model.load_weights(template + 'main.h5')
        model.policy_head.load_weights(template + 'policy.h5')
        model.value_head.load_weights(template + 'value.h5')

    sarsdv = []
    pbar = tqdm()

    rew = np.zeros((num_envs, 1))

    steps = 0
    num_eps = 0
    steps_since_last_test = 0
    while True:
        try:
            if not test_only:
                state, num_steps, num_eps = run_train_step(state=state, rew=rew, pbar=pbar, envs=envs, model=model, num_actions=num_actions, batch_size=batch_size, steps=steps, num_envs=num_envs, num_eps=num_eps, viz=viz_training)
                steps += num_steps
                steps_since_last_test += num_steps
                if steps_since_last_test >= 500000:
                    folder = run_name
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    model.model.save(f'{folder}/{steps}_main.h5')
                    model.policy_head.save(f'{folder}/{steps}_policy.h5')
                    model.value_head.save(f'{folder}/{steps}_value.h5')

            if (steps_since_last_test >= 500000) or test_only:
                steps_since_last_test = 0
                trew, tscore = run_test_step(model, test_env, 1/30.0)
                if not test_only:
                    wandb.log({'test_reward': trew, 'test_score': tscore}, step=steps)

            if not summaries_done:
                print(model.policy_head.summary())
                print(model.value_head.summary())
                summaries_done = True
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    argh.dispatch_command(main)
