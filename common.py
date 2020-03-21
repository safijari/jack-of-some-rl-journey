import numpy as np
from dataclasses import dataclass

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
    def __init__(self, env_factory, num_envs, pytorch=False, num_viz_train=0, viz_test=False, test_delay=0.01):
        self.envs = [env_factory() for _ in range(num_envs)]
        self.test_env = env_factory()
        self._p = lambda im: permute_axes_and_prep_image(im, pytorch)
        self.state = np.stack([self._p(env.reset()) for env in self.envs])
        self.num_viz_train = num_viz_train
        self.viz()

    def viz(self):
        for env in self.envs[:self.num_viz_train]:
            env.render()

    def apply_actions(self, actions):
        next_state, rewards, dones_list, info_dicts = zip(*[env.step(a) for env, a in zip(self.envs, actions)])
        next_state = np.stack([self._p(s) for s in next_state])
        rewards = np.expand_dims(np.stack(rewards), -1)
        dones = np.expand_dims(np.stack(dones_list), -1)

        out_state = self.state
        self.state = next_state
        for i, env in enumerate(self.envs):
            if dones_list[i]:
                self.state[i] = self._p(env.reset())

        self.viz()

        return out_state, rewards, dones, info_dicts


def compute_gae(next_value, rewards, dones, values, gamma=0.998, lmbda=0.99):
    masks = [1 - d for d in dones]
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1].detach().numpy() * masks[step] - values[step].detach().numpy()
        gae = delta + gamma * lmbda * masks[step] * gae
        returns.append(gae + values[step].detach().numpy())
    return list(reversed(returns))


if __name__ == '__main__':
    import gym
    from snake_gym import *
    env_fac = lambda: gym.make('snakenv-v0', gs=10, main_gs=12, num_fruits=1)
    m = EnvManager(env_fac, 4, True, 2)
    import ipdb; ipdb.set_trace()
