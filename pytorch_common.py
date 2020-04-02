import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from common import EnvManager, compute_gae
import gym
from snake_gym import SnakeEnv
import vizdoomgym
import wandb
import argh


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def ppo_iter(
    mini_batch_size, states, actions, log_probs, returns, advantage, r_states, device
):
    batch_size = states.size(0)
    for i in range(batch_size // mini_batch_size):
        out_rstates = (
            torch.zeros((1, 1))
            if isinstance(r_states, list)
            else r_states[i * mini_batch_size : (i + 1) * mini_batch_size, :].to(device)
        )
        yield states[i * mini_batch_size : (i + 1) * mini_batch_size, :].to(
            device
        ), actions[i * mini_batch_size : (i + 1) * mini_batch_size, :].to(
            device
        ), log_probs[
            i * mini_batch_size : (i + 1) * mini_batch_size, :
        ].to(
            device
        ), returns[
            i * mini_batch_size : (i + 1) * mini_batch_size, :
        ].to(
            device
        ), advantage[
            i * mini_batch_size : (i + 1) * mini_batch_size, :
        ].to(
            device
        ), out_rstates


class CuriosityTracker(nn.Module):
    def __init__(self, input_shape, num_hidden=512, device="cuda"):
        super(CuriosityTracker, self).__init__()
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.convs = nn.Sequential(
            init_(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.rand(input_shape).unsqueeze(0)
            x = self.convs(x)

        num_fc = x.view(1, -1).shape[1]

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.head = nn.Sequential(init_(nn.Linear(num_fc, num_hidden)), nn.ReLU(),)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.device = device

    def forward(self, x):  # hxs is size [Nbatch, 512], must be 0 at start of episode
        latent_ = self.convs(x / 255)
        latent = latent_.view(x.shape[0], -1)

        return self.head(latent)


class VisualAgentPPO(nn.Module):
    def __init__(
        self,
        input_shape,
        num_actions,
        num_hidden=512,
        device="cuda",
        smaller=False,
        recurrent=1024,
    ):
        super(VisualAgentPPO, self).__init__()
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        if not smaller:
            self.convs = nn.Sequential(
                init_(nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, kernel_size=4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 128, kernel_size=4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 256, kernel_size=3, stride=1)),
                nn.ReLU(),
            )
        else:
            self.convs = nn.Sequential(
                init_(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
            )

        with torch.no_grad():
            x = torch.rand(input_shape).unsqueeze(0)
            x = self.convs(x)

        num_fc = x.view(1, -1).shape[1]

        if recurrent:
            self._recurrent = recurrent
            self.gru = nn.GRU(num_fc, recurrent)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)
            num_fc = recurrent

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.policy = nn.Sequential(
            init_(nn.Linear(num_fc, num_hidden)),
            nn.ReLU(),
            init_(nn.Linear(num_hidden, num_actions)),
        )

        self.value = nn.Sequential(
            init_(nn.Linear(num_fc, num_hidden)),
            nn.ReLU(),
            init_(nn.Linear(num_hidden, 1)),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    def forward(
        self, x, hxs
    ):  # hxs is size [Nbatch, 512], must be 0 at start of episode
        latent_ = self.convs(x / 255)
        latent = latent_.view(x.shape[0], -1)

        if self._recurrent:
            # x will be [Nbatch, latent_size]
            latent, rnn_hxs = self.gru(latent.unsqueeze(0), hxs.unsqueeze(0))
            latent = latent.squeeze()
            rnn_hxs = rnn_hxs.squeeze()

        policy = self.policy(latent)
        value = self.value(latent)

        return (
            torch.distributions.categorical.Categorical(logits=policy),
            value,
            rnn_hxs,
        )

    def ppo_update_generator(
        self, generator, mini_batch_size, ppo_epochs, clip_param=0.2,
    ):
        model = self
        optimizer = self.optimizer
        final_loss = 0
        factor_loss = 0
        fcritic_loss = 0
        fentropy_loss = 0
        final_loss_steps = 0
        for ii in tqdm(range(ppo_epochs)):
            for state, action, old_log_probs, return_, advantage, r_state in generator(
                mini_batch_size
            ):
                dist, value, _ = model(state, r_state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action.view(-1)).unsqueeze(1)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                optimizer.zero_grad()
                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy
                loss.backward()
                optimizer.step()
                final_loss += loss.detach().item()
                factor_loss += actor_loss.detach().item()
                fcritic_loss += critic_loss.detach().item()
                fentropy_loss += entropy.detach().item()
                final_loss_steps += 1

        return (
            final_loss / final_loss_steps,
            factor_loss / final_loss_steps,
            fcritic_loss / final_loss_steps,
            fentropy_loss / final_loss_steps,
        )

    def ppo_update(
        self,
        ppo_epochs,
        mini_batch_size,
        states,
        actions,
        log_probs,
        returns,
        advantages,
        r_states,
        clip_param=0.2,
    ):
        model = self
        optimizer = self.optimizer
        final_loss = 0
        factor_loss = 0
        fcritic_loss = 0
        fentropy_loss = 0
        final_loss_steps = 0
        for _ in tqdm(range(ppo_epochs)):
            for state, action, old_log_probs, return_, advantage, r_state in ppo_iter(
                mini_batch_size,
                states,
                actions,
                log_probs,
                returns,
                advantages,
                r_states,
                self.device,
            ):
                dist, value, _ = model(state, r_state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action.view(-1)).unsqueeze(1)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                optimizer.zero_grad()
                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                loss.backward()
                optimizer.step()
                final_loss += loss.detach().item()
                factor_loss += actor_loss.detach().item()
                fcritic_loss += critic_loss.detach().item()
                fentropy_loss += entropy.detach().item()
                final_loss_steps += 1

        return (
            final_loss / final_loss_steps,
            factor_loss / final_loss_steps,
            fcritic_loss / final_loss_steps,
            fentropy_loss / final_loss_steps,
        )

    def save(self, path):
        # self.cpu()
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        # self.to(self.device)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def _t(l, fl=True):
    if fl:
        return torch.cat([torch.FloatTensor(i) for i in l])
    else:
        return torch.cat([torch.Tensor(i) for i in l])
