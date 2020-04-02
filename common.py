import numpy as np
import time
import torch
from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor


def permute_axes_and_prep_image(im, pytorch=False):
    assert len(im.shape) < 4
    if len(im.shape) == 2:
        channel_axis = 0 if pytorch else -1
        return np.expand_dims(im, channel_axis)
    else:
        if not pytorch:
            return im
        else:
            return np.transpose(im, (2, 0, 1))


class EnvManager:
    def __init__(
        self,
        env_factory,
        num_envs,
        pytorch=False,
        num_viz_train=0,
        viz_test=False,
        test_delay=0.01,
        reward_mult=1.0,
        skip=1,
    ):
        self.envs = [env_factory() for _ in range(num_envs)]
        self.test_env = env_factory()
        self._p = lambda im: permute_axes_and_prep_image(im, pytorch)
        self.reward_mult = reward_mult
        self.state = np.stack([self._p(env.reset()) for env in self.envs])
        self.num_viz_train = num_viz_train
        self.skip = skip
        self.ex = ThreadPoolExecutor(num_envs)
        self.viz()

    def viz(self):
        for env in self.envs[: self.num_viz_train]:
            env.render()
            # time.sleep(0.1)

    def apply_actions(self, actions):
        for i in range(self.skip):
            # next_state, rewards, dones_list, info_dicts = zip(
            #     *[env.step(a) for env, a in zip(self.envs, actions)]
            # )
            next_state, rewards, dones_list, info_dicts = zip(
                *(self.ex.map(lambda x: x[0].step(x[1]), zip(self.envs, actions)))
            )

            self.viz()
        next_state = np.stack([self._p(s) for s in next_state])
        rewards = np.expand_dims(np.stack(rewards), -1) * self.reward_mult
        dones = np.expand_dims(np.stack(dones_list), -1)

        out_state = self.state
        self.state = next_state
        for i, env in enumerate(self.envs):
            if dones_list[i]:
                self.state[i] = self._p(env.reset())

        return out_state, rewards, dones, info_dicts


def compute_gae(next_value, rewards, dones, values, gamma=0.999, lmbda=0.98):
    masks = [1 - d for d in dones]
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1].detach().numpy() * masks[step]
            - values[step].detach().numpy()
        )
        gae = delta + gamma * lmbda * masks[step] * gae
        returns.append(gae + values[step].detach().numpy())
    return list(reversed(returns))


class RolloutStorage:
    """
    fixed sized storage for generating rollouts
    """

    def __init__(self, num_steps, num_envs, obs_shape, num_actions, recurrent_size):
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape)
        self.recurrent_states = torch.zeros(num_steps + 1, num_envs, recurrent_size)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        self.actions = torch.zeros(num_steps, num_envs, 1)
        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.num_actions = num_actions
        self.step = 0

    def insert(
        self,
        obs,
        recurrent_state,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_states[self.step + 1].copy_(recurrent_state)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_states[0].copy_(self.recurrent_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma=0.99, lda=0.95):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * lda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def generate(self, mini_batch_size):
        # first dim is time, second dim is envs
        total = (self.obs.shape[0] - 1) * self.obs.shape[1]

        def git(tensor, i, obs=False):
            if obs:
                return tensor.view(-1, *tensor.size()[2:])[i : i + mini_batch_size]
            return tensor.view(-1, tensor.size(-1))[i : i + mini_batch_size]

        for i in range(0, total, mini_batch_size):  # random?
            obs_batch = git(self.obs[:-1], i, True)
            recur_batch = git(self.recurrent_states[:-1], i)
            actions_batch = git(self.actions, i)
            value_preds_batch = git(self.value_preds[:-1], i)
            return_batch = git(self.returns[:-1], i)
            masks_batch = git(self.masks[:-1], i)
            old_action_log_probs_batch = git(self.action_log_probs, i)

            try:
                yield obs_batch.cuda(), actions_batch.cuda(), old_action_log_probs_batch.cuda(), return_batch.cuda(), (
                    return_batch - value_preds_batch
                ).cuda(), recur_batch.cuda()
            except Exception:
                import ipdb

                ipdb.set_trace()
