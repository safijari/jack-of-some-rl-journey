import gym
from gym.envs.classic_control import rendering
import os
import numpy as np
from gym.utils import play
from gym import spaces
from gym import error, spaces, utils
from gym.utils import seeding
from snake import Env, SnakeState, INIT_TAIL_SIZE
import random
import time
import cv2


KEYWORD_TO_KEY = {
    (ord('j'), ): 1,
    (ord('l'), ): 2,
}

action_map = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

reward_map = {
    SnakeState.OK: -0.001,
    SnakeState.ATE: 1,
    SnakeState.DED: -1,
    SnakeState.WON: 1
}


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, gs=10, main_gs=10, num_fruits=1, action_map=None):
        super(SnakeEnv, self).__init__()
        self.env = Env(gs, main_gs=main_gs, num_fruits=num_fruits)
        self.viewer = None
        self.action_map = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }
        if action_map is not None:
            self.action_map = action_map

        self.action_space = spaces.Discrete(len(self.action_map.keys()))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.env.gs, self.env.gs, 3),
            dtype=np.uint8)
        self.idx = 0
        self.total_score = 0
        self.vis = False

    def step(self, action):
        enum = self.env.update(self.action_map[action])

        rew = reward_map[enum]

        is_done = (enum in [SnakeState.DED, SnakeState.WON])
        info_dict = {}
        if is_done:
            info_dict['score'] = len(self.env.snake.tail) - INIT_TAIL_SIZE

        return np.expand_dims(self.env.to_image().astype('float32'), -1), rew, is_done, info_dict

    @property
    def dist(self):
        return self.env.fruit_loc[0].dist(self.env.snake.head)

    def reset(self):
        self.vis = os.path.exists('/tmp/vis')
        self.idx = 0
        self.total_score = 0
        self.env.reset()
        self.last_dist = self.dist
        return np.expand_dims(self.env.to_image().astype('float32'), -1)

    def render(self, mode='human', close=False):
        im = self.env.to_image()
        if mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            im = self.env.to_image(True)

            im = cv2.resize(im, (640, 640), interpolation=0)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            self.viewer.imshow(im)
            return self.viewer.isopen
        elif mode == 'jack':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            self.viewer.imshow(cv2.resize(self.env.to_image(), (640, 640), interpolation=0))
            return self.viewer.isopen
        else:
            return np.expand_dims(cv2.resize(im[:,:,0], (84, 84), interpolation=0), -1)

try:
    gym.envs.register(id="snakenv-v0", entry_point='snake_gym:SnakeEnv')
except Exception:
    print('already done?')

if __name__ == '__main__':
    action_map = {
        0: None,
        1: 'up',
        2: 'down',
        3: 'left',
        4: 'right'
    }

    KEYWORD_TO_KEY = {
        (ord('i'), ): 1,
        (ord('j'), ): 3,
        (ord('k'), ): 2,
        (ord('l'), ): 4,
    }

    def callback(obs_t, obs_tp1, action, rew, done, info):
        print(rew)

    env = gym.make('snakenv-v0', gs=20, main_gs=40, action_map=action_map, num_fruits=1)
    play.keys_to_action = KEYWORD_TO_KEY
    play.play(env, fps=15, keys_to_action=KEYWORD_TO_KEY, callback=callback)
