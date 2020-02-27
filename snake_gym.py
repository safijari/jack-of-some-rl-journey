import gym
import traceback
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
    (ord('i'), ): 0,
    (ord('k'), ): 1,
    (ord('j'), ): 2,
    (ord('l'), ): 3,
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
    def __init__(self, gs=10, human_mode_sleep=0.05, seed=None, use_running_log=False, allow_viz=False):
        super(SnakeEnv, self).__init__()
        self.env = Env(gs, seed=seed)
        self.human_mode_sleep = human_mode_sleep
        self.running_log = []
        self.use_running_log = use_running_log
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.env.gs, self.env.gs, 3),
            dtype=np.uint8)
        self.idx = 0
        self.total_score = 0
        self.vis = False
        self.allow_viz = allow_viz

    def step(self, action):
        # self.running_log.append(self.env.to_dict())
        self.idx += 1
        enum = self.env.update(action_map[action])
        done = (enum in [SnakeState.DED, SnakeState.WON])
        rew = reward_map[enum]
        self.total_score += rew
        if self.vis and self.allow_viz:
            self.render()
        # if done:
        #     print(self.idx, self.total_score)
        return self.render('other'), rew, done, {}

    def reset(self):
        self.vis = os.path.exists('/tmp/vis')
        self.idx = 0
        self.total_score = 0
        self.env.reset()
        if self.running_log and self.use_running_log and random.random() > 0.75:
            idx = random.randint(0, len(self.running_log) - 1)
            # print(f"keeping {idx + 1} out of {len(self.running_log)} past transitions")
            self.env.from_dict(self.running_log[idx])
            self.running_log = self.running_log[:idx]
        else:
            self.running_log = []
        return self.render('other')

    def render(self, mode='human', close=False):
        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            im = self.env.to_image(True)

            im = cv2.resize(im, (640, 640), interpolation=0)
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            self.viewer.imshow(im)
            time.sleep(self.human_mode_sleep)
            return self.viewer.isopen
        elif mode == 'other':
            # im = self.env.to_image()
            return self.env.to_image()
            # return np.expand_dims(cv2.resize(im[:,:,0], (40, 40), interpolation=0), -1)
        elif mode == 'jack':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=640)
                self.viewer.height = 640
                self.viewer.width = 640

            self.viewer.imshow(cv2.resize(self.env.to_image(), (640, 640), interpolation=0))
            return self.viewer.isopen
        else:
            im = self.env.to_image()
            return cv2.resize(im[:,:,::-1], (640, 640), interpolation=0)

try:
    gym.envs.register(id="snakenv-v0", entry_point='snake_gym:SnakeEnv')
except Exception:
    traceback.print_exc()
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
