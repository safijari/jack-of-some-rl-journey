import gym
import os
import numpy as np
from gym.utils import play
from gym import spaces
from gym import error, spaces, utils
from gym.utils import seeding
from snake import Env, SnakeState
import random
import time
import cv2


KEYWORD_TO_KEY = {
    (ord('i'), ): 1,
    (ord('k'), ): 2,
    (ord('j'), ): 3,
    (ord('l'), ): 4,
}

action_map = {
    0: None,
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}


reward_map = {
    SnakeState.OK: -0.01,
    SnakeState.ATE: 1,
    SnakeState.DED: -1,
    SnakeState.WON: 1
}


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, gs=10, human_mode_sleep=0.05, seed=None):
        super(SnakeEnv, self).__init__()
        self.env = Env(gs, seed=seed)
        self.human_mode_sleep = human_mode_sleep
        self.running_log = []
        self.use_running_log = True
        if 'USE_RUNNING_LOG' in os.environ and os.environ['USE_RUNNING_LOG'] == 'false':
            self.use_running_log = False
        self.viewer = None
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.env.gs, self.env.gs, 3),
            dtype=np.uint8)
        self.idx = 0
        self.total_score = 0

    def step(self, action):
        self.running_log.append(self.env.to_dict())
        self.idx += 1
        enum = self.env.update(action_map[action])
        done = (enum in [SnakeState.DED, SnakeState.WON])
        rew = reward_map[enum]
        self.total_score += rew
        # if done:
        #     print(self.idx, self.total_score)
        return self.env.to_image(), rew, done, {}

    def reset(self):
        self.idx = 0
        self.total_score = 0
        self.env.reset()
        if self.running_log and self.use_running_log:
            idx = random.randint(0, len(self.running_log) - 1)
            # print(f"keeping {idx + 1} out of {len(self.running_log)} past transitions")
            self.env.from_dict(self.running_log[idx])
            self.running_log = self.running_log[:idx]
        return self.env.to_image()

    def render(self, mode='human', close=False):
        im = self.env.to_image()
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            im = self.env.to_image(True)

            im = cv2.resize(im, (640, 640), interpolation=0)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            self.viewer.imshow(im)
            time.sleep(self.human_mode_sleep)
            return self.viewer.isopen
            # return im
        elif mode == 'jack':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            self.viewer.imshow(cv2.resize(self.env.to_image(), (640, 640), interpolation=0))
            return self.viewer.isopen
        else:
            return cv2.cvtColor(cv2.resize(im, (640, 640), interpolation=0), cv2.COLOR_GRAY2BGR)

try:
    gym.envs.register(id="snakenv-v0", entry_point='snake_gym:SnakeEnv')
except Exception:
    print('already done?')

if __name__ == '__main__':
    def callback(obs_t, obs_tp1, action, rew, done, info):
        try:
            callback.rew += rew
        except Exception:
            callback.rew = rew
        print(callback.rew)

    env = gym.make('snakenv-v0')
    # env = SnakeEnv()
    play.keys_to_action = KEYWORD_TO_KEY
    play.play(env, fps=5, keys_to_action=KEYWORD_TO_KEY, callback=callback)
