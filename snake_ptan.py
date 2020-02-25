 #!/usr/bin/env python -W ignore::DeprecationWarning
import ptan
from tensorboardX import SummaryWriter
import argh
import gym
from snake_gym import *
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

def conv_cell(inchannels, outchannels, kernel, stride=1):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size=kernel, stride=stride),
        nn.ReLU()
    )

class Net(nn.Module):
    def __init__(self, grid_size, n_actions, channels=1):
        super(Net, self).__init__()
        self.gs = grid_size
        self.convs = nn.Sequential(
            conv_cell(channels, 64, 3, 1),
            conv_cell(64, 128, 3, 1),
            conv_cell(128, 128, 3, 2),
            )

        # make up a 1 item batch and put it through the convs
        convs_result = self.convs(torch.zeros(1, *(channels, grid_size, grid_size)))
        # take the result's size and prod it all together, that's the flattened size
        out_size = int(np.prod(convs_result.size()))

        hs = 512

        self.fc = nn.Sequential(
            nn.Linear(out_size, hs),
            nn.ReLU(),
            nn.Linear(hs, n_actions)
            )

    def forward(self, x: torch.Tensor):
        fx = x.float() / 255
        # [0] is the batch dim, so the below resizes the conv out to [batch, whatevs]
        fcin = self.convs(fx).view(fx.size()[0], -1)
        return self.fc(fcin)


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask.bool()] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


def run_test(net, env, max_steps=200, visualize=False):
    idx = 0
    state = torch.Tensor(env.reset()._frames)
    action = torch.argmax(net(state))
    total_reward = 0
    done = False
    for i in range(max_steps):
        if visualize:
            env.render()
            time.sleep(0.02)
        state, reward, done, _ = env.step(action.item())
        state = torch.Tensor(state._frames)
        action = torch.argmax(net(state))
        total_reward += reward
        if done:
            break

    if visualize:
        env.render()

    return i, total_reward, done


def main(run_name, shape=4, winsize=1, test=False, num_max_test=200, visualize_training=False, start_steps=0, randseed=None, human_mode_sleep=0.02, device='cpu', gamma=0.99, tgt_net_sync=20000):
    INPUT_SHAPE = (shape, shape)
    WINDOW_LENGTH = winsize

    try:
        randseed = int(randseed)
        print(f"set seed to {randseed}")
    except Exception:
        print(f"failed to intify seed of {randseed}, making it None")
        randseed = None
    env = gym.make('snakenv-v0', gs=shape, seed=randseed, human_mode_sleep=human_mode_sleep)
    env = ptan.common.wrappers.ImageToPyTorch(env)
    env = ptan.common.wrappers.FrameStack(env, winsize)

    test_env = gym.make('snakenv-v0', gs=shape, seed=randseed, human_mode_sleep=human_mode_sleep)
    test_env = ptan.common.wrappers.ImageToPyTorch(test_env)
    test_env = ptan.common.wrappers.FrameStack(test_env, winsize)

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    writer = SummaryWriter(comment=run_name)

    net = Net(shape, env.action_space.n, channels=winsize).to(device)
    # m = torch.load('./4by4_w1_pytorch_first/1000000.pth')
    # net.load_state_dict(m())
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(1.0)
    epsilon_tracker = ptan.actions.EpsilonTracker(selector, 1.0, 0.05, 2500000)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma, steps_count=1)

    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, 500000, 0.6)
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    replay_initial = 5000


    for i in range(5000000):
        buffer.populate(1)

        if len(buffer) < replay_initial:
            continue

        tidx = i - replay_initial
        epsilon_tracker.frame(tidx)

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(32, beta=0.4) # beta???
        loss_v, sample_prios_v = calc_loss(
            batch, batch_weights, net, tgt_net.target_model, gamma, device=device)

        writer.add_scalar('loss', loss_v.item(), tidx)
        writer.add_scalar('eps', selector.epsilon, tidx)

        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

        if tidx % 1000 == 0:
            visualize = os.path.exists('/tmp/vis')
            epsteps, rew, done = run_test(net, test_env, visualize=visualize)
            writer.add_scalar('test_reward', rew, tidx)
            writer.add_scalar('test_episode_len', epsteps + 1, tidx)
            m = exp_source.pop_rewards_steps()[-1]
            writer.add_scalar('explore_reward', m[0], tidx)
            writer.add_scalar('explore_episode_len', m[1], tidx)

        if tidx % tgt_net_sync == 0:
            if tidx > 0:
                tgt_net.sync()
                if not os.path.exists(run_name):
                    os.makedirs(run_name)
                torch.save(net.state_dict, os.path.join(run_name, f"{tidx}.pth"))

if __name__ == '__main__':
    argh.dispatch_command(main)
