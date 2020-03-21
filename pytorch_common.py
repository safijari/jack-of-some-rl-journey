import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from common import EnvManager, compute_gae
import gym
from snake_gym import SnakeEnv
import wandb


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage, r_states, device):
    batch_size = states.size(0)
    for i in range(batch_size // mini_batch_size):
        out_rstates = torch.zeros((1,1)) if isinstance(r_states, list) else r_states[i*mini_batch_size:(i+1)*mini_batch_size, :].to(device)
        yield states[i*mini_batch_size:(i+1)*mini_batch_size, :].to(device), actions[i*mini_batch_size:(i+1)*mini_batch_size, :].to(
            device
        ), log_probs[i*mini_batch_size:(i+1)*mini_batch_size, :].to(device), returns[i*mini_batch_size:(i+1)*mini_batch_size, :].to(
            device
        ), advantage[
            i*mini_batch_size:(i+1)*mini_batch_size, :
        ].to(
            device
        ), out_rstates


class SnakeModel(nn.Module):
    def __init__(self, input_shape, num_actions, num_hidden=512, device="cuda", smaller=False, recurrent=1024):
        super(SnakeModel, self).__init__()
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        if not smaller:
            self.convs = nn.Sequential(
                init_(nn.Conv2d(1, 64, kernel_size=8, stride=4)),
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
                init_(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
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
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
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
            init_(nn.Linear(num_hidden, 1))
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)
        self.device = device

    def forward(self, x, hxs):  # hxs is size [Nbatch, 512], must be 0 at start of episode
        latent_ = self.convs(x / 255)
        latent = latent_.view(x.shape[0], -1)

        if self._recurrent:
            # x will be [Nbatch, latent_size]
            latent, rnn_hxs = self.gru(latent.unsqueeze(0), hxs.unsqueeze(0))
            latent = latent.squeeze()
            rnn_hxs = rnn_hxs.squeeze()

        policy = self.policy(latent)
        value = self.value(latent)

        return torch.distributions.categorical.Categorical(logits=policy), value, rnn_hxs

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


def _t(l):
    return torch.cat([torch.FloatTensor(i) for i in l])


def main(device="cuda"):
    recurrent = True
    recurrent_size = 1024 if recurrent else 0
    wandb.init(project="snake-pytorch-ppo")
    env_fac = lambda: gym.make("snakenv-v0", gs=20, main_gs=22, num_fruits=1)
    num_envs = 4
    num_steps = 512
    m = EnvManager(env_fac, num_envs, pytorch=True, num_viz_train=0)
    s = m.state.shape[-1]
    model = SnakeModel((1, s, s), 4, device=device, recurrent=recurrent_size, smaller=True).to(device)

    idx = 0
    batch_num = 0
    episode_num = 0

    recurrent_state = torch.zeros((num_envs, recurrent_size))

    while True:
        states = []
        r_states  = []
        values = []
        rewards = []
        dones = []
        actions = []
        info_dicts = []
        log_probs = []
        scores = []

        with torch.no_grad():
            for i in range(num_steps):
                if recurrent:
                    r_states.append(recurrent_state.cpu())
                dist, v, recurrent_state = model(torch.FloatTensor(m.state).to(device), recurrent_state.to(device))
                idx += num_envs
                acts_normal = dist.sample()
                # acts_exploit = dist.logits.max(1).indices
                # acts = []
                # for i in range(len(acts_exploit)):
                #     if i % 2 == 0:
                #         acts.append(acts_normal[i])
                #     else:
                #         acts.append(acts_exploit[i])
                # acts = torch.FloatTensor(acts).to(device)
                acts = acts_normal
                ost, r, d, idicts = m.apply_actions(acts.tolist())
                states.append(ost)
                rewards.append(r)
                dones.append(d)
                values.append(v)
                log_prob = dist.log_prob(acts)
                log_probs.append(log_prob)
                actions.append(acts)

                for i, dun in enumerate(d):
                    if dun:
                        recurrent_state[i] = 0

                if any(d):
                    episode_num += 1
                    scores.extend([idict["score"] for idict in idicts if "score" in idict])
                    # wandb.log({"episode_score": max(scores)}, step=idx)

            gae_ = compute_gae(
                model(torch.FloatTensor(m.state).to(device), recurrent_state.to(device))[1].cpu(),
                rewards,
                dones,
                [v.cpu() for v in values],
            )
            gae = _t(gae_)
            values = torch.cat(values)
            log_probs = torch.cat(log_probs).unsqueeze(-1)
            advantage = gae.to(device) - values
            actions = torch.cat(actions).unsqueeze(-1)
            states = _t(states)
            if recurrent:
                r_states = torch.cat(r_states)

        if os.path.exists("/tmp/debug_jari"):
            try:
                os.remove("/tmp/debug_jari")
            except Exception:
                pass
            import ipdb; ipdb.set_trace()

        loss, actor_loss, critic_loss, entropy_loss = model.ppo_update(
            2, min(num_envs*num_steps, 2048), states, actions, log_probs, gae, advantage, r_states
        )
        batch_num += 1
        score = 0 if not scores else max(scores)
        wandb.log(
            {
                "loss": loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "score": score,
                "steps": idx,
                "episodes": episode_num
            },
            step=batch_num,
        )


if __name__ == "__main__":
    main()
