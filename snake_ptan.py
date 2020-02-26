import math
import ptan
from tensorboardX import SummaryWriter
import wandb
import argh
import gym
from snake_gym import *
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

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
    def __init__(self, grid_size, n_actions, channels=1, use_noisy_linear=False):
        super(Net, self).__init__()
        if grid_size >= 10:
            self.convs = nn.Sequential(
                conv_cell(channels, 64, 3, 1),
                conv_cell(64, 128, 3, 1),
                conv_cell(128, 128, 3, 2),
                )
        else:
            self.convs = nn.Sequential(
                conv_cell(channels, 128, 3, 1))

        # make up a 1 item batch and put it through the convs
        convs_result = self.convs(torch.zeros(1, *(channels, grid_size, grid_size)))
        # take the result's size and prod it all together, that's the flattened size
        out_size = int(np.prod(convs_result.size()))

        hs = 256

        if use_noisy_linear:
            self.fc = nn.Sequential(
                NoisyLinear(out_size, hs),
                nn.ReLU(),
                NoisyLinear(hs, n_actions)
                )
        else:
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

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask.bool()] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean()


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


def main(run_name, shape=10, winsize=4, num_max_test=1000, randseed=None, human_mode_sleep=0.02, device='cpu', gamma=0.99, tgt_net_sync=5000):

    INPUT_SHAPE = (shape, shape)
    WINDOW_LENGTH = winsize

    lr = 0.001
    replay_size = 500000

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

    net = Net(shape, env.action_space.n, channels=winsize).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    batch_size = 32

    wandb.init(project='snake-rl-ptan', name=run_name, config={
        'lr': lr,
        'replay_size': replay_size,
        'net': str(net),
        'randseed': randseed,
        'gamma': gamma,
        'tgt_net_sync': tgt_net_sync,
        'shape': shape,
        'winsize': winsize,
        'batch_size': batch_size,
    })

    wandb.watch(net)
    wandb.watch(tgt_net.target_model)

    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma, steps_count=1)

    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, replay_size)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    replay_initial = int(replay_size/10)

    wandb.init(project='snake-rl-ptan', name=run_name)


    for i in range(5000000):
        buffer.populate(1)

        if len(buffer) < replay_initial:
            continue

        tidx = i - replay_initial

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_v = calc_loss(
            batch, net, tgt_net.target_model, gamma, device=device)

        loss_v.backward()
        optimizer.step()

        try:
            m = exp_source.pop_rewards_steps()[-1]
            wandb.log({
                'loss': loss_v.item(),
                'explore_reward': m[0],
                'explore_episode_len': m[1]
            }, step=tidx)
        except Exception:
            pass

        if tidx % 5000 == 0:
            visualize = os.path.exists('/tmp/vis')
            epsteps, rew, done = run_test(net, test_env, visualize=visualize)
            wandb.log({
                'loss': loss_v.item(),
                'test_reward': m[0],
                'test_episode_len': m[1]
            }, step=tidx)

        if tidx % tgt_net_sync == 0:
            if tidx > 0:
                tgt_net.sync()
                if not os.path.exists(run_name):
                    os.makedirs(run_name)
                torch.save(net.state_dict, os.path.join(run_name, f"{tidx}.pth"))


def main_reinforce(run_name, shape=4, winsize=1, num_max_test=1000, randseed=None, human_mode_sleep=0.02, device='cpu', gamma=0.99):

    INPUT_SHAPE = (shape, shape)
    WINDOW_LENGTH = winsize

    lr = 0.001

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

    net = Net(shape, env.action_space.n, channels=winsize).to(device)

    max_batch_episodes = 100

    wandb.init(project='snake-rl-reinforce', name=run_name, config={
        'lr': lr,
        'net': str(net),
        'randseed': randseed,
        'gamma': gamma,
        'shape': shape,
        'winsize': winsize,
        'max_batch_episodes': max_batch_episodes,
    })

    wandb.watch(net)

    # agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)
    agent = ptan.agent.PolicyAgent(net, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma, steps_count=1)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    wandb.init(project='snake-rl-ptan', name=run_name)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards, gamma))
            cur_rewards.clear()
            batch_episodes += 1


        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            wandb.log({
                'reward': reward,
                'reward_100': mean_rewards,
                'episodes': done_episodes
            }, step=step_idx)

        if batch_episodes < max_batch_episodes:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_qvals_v)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        wandb.log({
            'loss': loss_v.item(),
        }, step=step_idx)
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()


        if step_idx % 10000 == 0:
            if not os.path.exists(run_name):
                os.makedirs(run_name)
            torch.save(net.state_dict, os.path.join(run_name, f"reinforce_{step_idx}.pth"))

if __name__ == '__main__':
    argh.dispatch_commands([main, main_reinforce])
