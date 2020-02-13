import os
import cv2
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from typing import Any
import time
import random
from threeviz.api import plot_3d, plot_pose, plot_line_seg
import cv2
from random import seed
from maze_nn import create_maze_solving_network, predict_on_model, preprocess_image, transfer_weights_partially, add_rl_loss_to_network, make_intermediate_models
from collections import deque
import tensorflow as tf
import argh
import tensorflow.keras as kr

"""
- States
  - 0 means free
  - -1 mean not traversable
  - 1 means goal?
"""

def anneal_probability(itr, maxitr, start_itr, start_prob):
    m = (1-start_prob)/(maxitr-start_itr)
    b = start_prob
    return m*(itr - start_itr) + b

@dataclass
class SingleStep:
    st: Any
    stn: Any
    at: int
    rt: float
    done: bool


class Agent:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return (self.i, self.j)

    def vmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i + direction, self.j)

    def hmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i, self.j + direction)

    def __repr__(self):
        return str(self.loc)

class QLearning:
    def __init__(self, num_states, num_actions, lr=0.1, discount_factor=0.99):
        self.q = np.zeros((num_states, num_actions))
        self.a = lr
        self.g = discount_factor

    def update(self, st, at, rt, st1):
        q = self.q
        a = self.a
        g = self.g
        q[st, at] = (1 - a)*q[st, at] + a * (rt + g * np.max(q[st1]))


class Maze:
    def __init__(self, rows=4, columns=4):
        self.env = np.zeros((rows, columns))
        self.mousy = Agent(0, 0)

    def randomize_agent(self):
        X, Y = np.where(self.env == 0)
        i = random.randint(0, len(X)-1)
        # return X[i], Y[i]
        self.mousy.i = X[i]
        self.mousy.j = Y[i]

    def reset(self):
        self.mousy.i = 0
        self.mousy.j = 0

    def in_bounds(self, i, j):
        nr, nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc

    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)

    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a)

    @property
    def all_actions(self):
        a = self.mousy
        return [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1),
        ]

    def apply_action(self, idx):
        moves = self.all_actions
        assert idx >= 0 and idx < len(moves), f"Index {idx} is not valid for picking a move"
        move = moves[idx]
        score = -0.01
        win_score = 1
        death_score = -1
        if not self.is_valid_new_agent(move):
            return score, False
        self.do_a_move(move)
        if self.has_won():
            return win_score, True
        if self.has_died():
            return death_score, True

        return score, False

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "Mousy can't go there"
        self.mousy = a
        return 10 if self.has_won() else -0.1

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

    def has_died(self):
        a = self.mousy
        return self.env[a.i, a.j] == -1

    def has_ended(self):
        return self.has_won() or self.has_died()

    def visualize(self):
        nr, nc = self.env.shape
        z = -0.1
        a = self.mousy
        plot_line_seg(0, 0, z, nr, 0, z, 'e1', size=0.2, color='red')
        plot_line_seg(0, 0, z, 0, nc, z, 'e2', size=0.2, color='red')
        plot_line_seg(0, nc, z, nr, nc, z, 'e3', size=0.2, color='red')
        plot_line_seg(nr, 0, z, nr, nc, z, 'e4', size=0.2, color='red')
        plot_3d(*get_midpoint_for_loc(a.i, a.j), z, 'mousy', color='blue', size=1)
        plot_3d(*get_midpoint_for_loc(nr-1,nc-1), z, 'goal', color='green', size=1)

        xarr, yarr = np.where(self.env == -1)
        plot_3d(xarr + 0.5, yarr + 0.5, [z]*len(xarr), 'obstacles', size=1.0)

    def to_image(self, image_shape=64):
        a = self.mousy
        e = self.env
        imout = np.expand_dims(np.ones_like(e)*255, -1).astype('uint8')
        imout = np.dstack((imout, imout, imout))
        imout[e==-1, :] = 0
        imout[a.i, a.j, :-1] = 0
        imout[-1, -1, ::2] = 0
        return cv2.resize(imout, (image_shape, image_shape), interpolation=cv2.INTER_NEAREST)

def get_midpoint_for_loc(i, j):
    return i + 0.5, j + 0.5

def make_test_maze(s=4):
    # seed(9001)
    m = Maze(s,s)
    e = m.env
    h, w = e.shape
    e[-1, -1] = 1
    for i in range(len(e)):
        for j in range(len(e[i])):
            if i in [0, h-1] and j in [0, w-1]:
                continue
            if random.random() < 0.3:
                e[i, j] = -1
    # seed(time.time())
    return m

def run_episode(m, model, eps, memory, verbose=False, max_steps=None):
    # if not memory:
    #     memory = []
    m.reset()
    # m.randomize_agent()
    final_score = 0

    itr = 0
    agents = []

    while not m.has_ended(): # and not m.has_died():
        itr += 1
        if max_steps and itr > max_steps:
            return memory
        # if random.random() > anneal_probability(i, max_episodes, switch_episodes, 0.5) or i < switch_episodes:
        if random.random() < eps:
            idx = random.randint(0, 3)
        else:
            idx = predict_on_model(m.to_image(), model, False)

        at = idx
        state = m.to_image(64)
        rt, _ = m.apply_action(at)
        next_state = m.to_image(64)
        final_score += rt

        if verbose:
            m.visualize()
            time.sleep(0.05)

        done = final_score if m.has_ended() else 0
        memory.append(SingleStep(st=state, stn=next_state, rt=rt, at=at, done=done))

    # print(f"finished episode with final score of {final_score} and in {itr} iterations")
    return memory

def main(experiment_name, fw, starting_weights=None):
    # side_len = 5
    folder = f'models/{experiment_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    g = 0.95
    mem_size = 50000
    batch_size = 128
    side_len = 6
    # memory = deque(maxlen=1000)

    if not starting_weights:
        model = create_maze_solving_network()
        target_model = create_maze_solving_network()
        target_model.set_weights(model.get_weights())
        starting_episode = 0
    else:
        model = tf.keras.models.load_model(starting_weights)
        target_model = tf.keras.models.load_model(starting_weights)
        starting_episode = int(os.path.splitext(os.path.basename(starting_weights))[0])

    train_model = add_rl_loss_to_network(model)
    eps = 1.0
    decay_factor = 0.9999

    memory = deque(maxlen=mem_size)

    def tbwrite(name, data, step):
        tf.summary.scalar(name, data=data, step=step)

    print("bootstrapping")
    while len(memory) < mem_size:
        side_len = random.randint(3, 6)
        m = make_test_maze(side_len)
        run_episode(m, target_model, eps, memory, False)
    print("done bootstrapping")

    for i in range(starting_episode, 1000000):
        side_len = random.randint(3, 6)
        m = make_test_maze(side_len)
        m.randomize_agent()
        run_episode(m, target_model, eps, memory, False)

        steps = random.sample(memory, min(batch_size, len(memory)))

        inputs = []
        outputs = []
        masks = []

        target_vectors = model.predict(np.stack([s.st for s in steps], 0)/255.0)
        fut_actions = target_model.predict(np.stack([s.stn for s in steps], 0)/255.0)

        for j, s in enumerate(steps):
            x = s.st
            r = s.rt
            # target_vector = predict_on_model(s.st, model, True)
            # fut_action = predict_on_model(s.stn, target_model, True)
            target_vector, fut_action = target_vectors[j].copy(), fut_actions[j].copy()
            target = r
            if not s.done:
                target += g * np.max(fut_action)
            target_vector[s.at] = target
            mask = target_vector.copy()*0
            mask[s.at] = 1

            inputs.append(preprocess_image(x, expand=False))
            outputs.append(target_vector)
            masks.append(mask)

        # model.fit(np.stack(inputs, 0), np.stack(outputs, 0), epochs=1)
        targets = np.stack(outputs, 0)
        masks = np.stack(masks, 0)
        hist = train_model.fit([np.stack(inputs, 0), targets, masks],
                               targets, epochs=1)
        tbwrite('eps', eps, i)
        eps *= decay_factor
        eps = max(eps, 0.1)
        for k, v in hist.history.items():
            tbwrite(k, v[0], i)

        if i % 50 == 0:
            side_len = random.randint(3, 6)
            m = make_test_maze(side_len)
            transfer_weights_partially(model, target_model, 1)
            target_model.save(f'{folder}/{i}.h5')
            m.visualize()
            idx = 0
            score = 0
            while not m.has_ended():
                time.sleep(0.1)
                _score, _ = m.apply_action(predict_on_model(m.to_image(64), target_model, False))
                score += _score
                m.visualize()
                idx += 1
                if idx > 20:
                    break
            tbwrite('test_score', score, i)
        dones = [m.done for m in memory if m.done][-100:]
        tbwrite('mean_score', sum(dones)/len(dones), i)

def run_training(experiment_name, starting_weights=None):
    logdir = "logs/scalars/" + experiment_name
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    main(experiment_name, file_writer, starting_weights)

def run_test(weights_path, side_len=4):
    model = tf.keras.models.load_model(weights_path)
    while True:
        m = make_test_maze(side_len)
        run_episode(m, model, 0, [], True, max_steps=25)

def rescale_image(im):
    im = (im - im.min())
    im = im/im.max()*255
    return im.astype('uint8')

if __name__ == '__main__':
    argh.dispatch_commands([run_training, run_test])
    # m = kr.models.load_model('models/my_model_64_img_50iter_weight_transfer_with_obstacles_3x3to5x5_randomized_128_batch/40000.h5')
    # models = make_intermediate_models(m)
    # m = make_test_maze(5)
    # im = preprocess_image(m.to_image(64))
    # cv2.imwrite('/tmp/start_image.jpg', rescale_image(im[0, :, :, :]))

    # res = [model.predict(im) for model in models]
    # for ii, r in enumerate(res):
    #     for i in range(r.shape[-1]):
    #         cv2.imwrite(f'/tmp/layer{ii + 1}_{i}.jpg', rescale_image(r[0, :, :, i]))
