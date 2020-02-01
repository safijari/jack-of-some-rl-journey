import numpy as np
from collections import namedtuple
import time
import random
from threeviz.api import plot_3d, plot_pose, plot_line_seg

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

    def reset(self):
        self.mousy.i = 0
        self.mousy.j = 0

    def state_for_agent(self, a):
        nr, nc = self.env.shape
        return a.i * nc + a.j

    def in_bounds(self, i, j):
        nr, nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc

    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)

    def agent_dient(self, a):
        return not self.env[a.i, a.j] == -1

    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a) and self.agent_dient(a)

    @property
    def all_actions(self):
        a = self.mousy
        return [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1),
        ]

    def compute_possible_moves(self):
        moves = self.all_actions
        return [(m, ii) for ii, m in enumerate(moves) if self.is_valid_new_agent(m)]

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "Mousy can't go there"
        self.mousy = a
        return 10 if self.has_won() else -0.1

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

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

def get_midpoint_for_loc(i, j):
    return i + 0.5, j + 0.5

def make_test_maze(s=4):
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
    return m

def main():
    s = 8
    q = QLearning(s**2, 4)

    go_ahead = False

    print("Ensure that you have safijari.github.io/threeviz open in chrome")
    print("and see if a maze shows up (if not just hit n and enter).")
    print("Cycle through the mazes until a viable one shows up and then enter y to continue.")

    while not go_ahead:
        m = make_test_maze(s)
        m.visualize()
        ctu = input()
        if ctu.lower() == 'n':
            continue
        go_ahead = True

    max_episodes = 200
    switch_episodes = 100

    for i in range(max_episodes):
        m.reset()
        final_score = 0

        itr = 0
        agents = []
        while not m.has_won():
            itr += 1
            if random.random() > anneal_probability(i, max_episodes, switch_episodes, 0.5) or i < switch_episodes:
                moves = m.compute_possible_moves()
                random.shuffle(moves)
                move, move_idx = moves[0]
            else:
                moves = m.all_actions
                s = m.state_for_agent(m.mousy)
                move_idx = np.argmax(q.q[s])
                move = moves[move_idx]

            at = move_idx
            st = m.state_for_agent(m.mousy)

            agents.append(m.mousy)

            score = m.do_a_move(move)
            final_score += score
            rt = score

            st1 = m.state_for_agent(m.mousy)

            q.update(st, at, rt, st1)

        print(f"finished episode with final score of {final_score} and in {itr} iterations")

    m.reset()
    m.visualize()
    while not m.has_won():
        time.sleep(0.1)
        s = m.state_for_agent(m.mousy)
        a_idx = np.argmax(q.q[s])
        m.do_a_move(m.all_actions[a_idx])
        m.visualize()

    m.visualize()

if __name__ == '__main__':
    main()
