import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    mini_batch_size,
    states,
    next_states,
    actions,
    log_probs,
    returns,
    advantage,
    r_states,
    device,
):
    batch_size = states.size(0)
    for i in range(batch_size // mini_batch_size):
        out_rstates = (torch.zeros((1, 1)) if isinstance(r_states, list) else
                       r_states[i * mini_batch_size:(i + 1) *
                                mini_batch_size, :].to(device))

        # vecst1s, forwards, backwards
        out_next_states = (None if next_states is None else
                           [item[i * mini_batch_size:(i + 1) *
                                 mini_batch_size, :].to(device) for item in next_states])

        yield states[i * mini_batch_size:(i + 1) * mini_batch_size, :].to(
            device), out_next_states, actions[
                i * mini_batch_size:(i + 1) *
                mini_batch_size, :].to(device), log_probs[
                    i * mini_batch_size:(i + 1) *
                    mini_batch_size, :].to(device), returns[
                        i * mini_batch_size:(i + 1) *
                        mini_batch_size, :].to(device), advantage[
                            i * mini_batch_size:(i + 1) *
                            mini_batch_size, :].to(device), out_rstates


def make_base_conv(input_shape, smaller=False):
    init_ = lambda m: init(
        m,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        nn.init.calculate_gain("relu"),
    )

    if not smaller:
        convs = nn.Sequential(
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
        convs = nn.Sequential(
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
        x = convs(x)

    flattened_size = x.view(1, -1).shape[1]

    return convs, flattened_size


class ICM(nn.Module):
    def __init__(self, input_shape, num_actions, smaller, feature_size=256):
        super(ICM, self).__init__()
        convs, num_fc = make_base_conv(input_shape, smaller)
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.convs = convs

        # converts st and st1 into features after conv
        self.featurizer = nn.Sequential(
            init_(nn.Linear(num_fc, feature_size)),
            nn.ReLU(),
        )

        # converts at (one hot) and st features into st1 features
        self.forward_model = nn.Sequential(
            init_(nn.Linear(feature_size + num_actions, feature_size)),
            nn.ReLU(),
        )

        # converts concatenated st and st1 into at (one hot)
        self.inverse_model = nn.Sequential(
            init_(nn.Linear(feature_size * 2, feature_size)),
            nn.ReLU(),
            init_(nn.Linear(feature_size, num_actions)),
            nn.Softmax(),
        )

    def forward(self, st, st1, at):  # this computes and sends out loss
        with torch.no_grad():
            stl = self.convs(st / 255)
            st1l = self.convs(st1 / 255)
            phst = self.featurizer(stl.view(stl.shape[0], -1))
            phst1 = self.featurizer(st1l.view(st1l.shape[0], -1))

        at_onehot = nn.functional.one_hot(at.squeeze(),
                                          self.num_actions).float()

        phst1_pred = self.forward_model(torch.cat([phst, at_onehot], 1))

        at_pred = self.inverse_model(torch.cat([phst, phst1], 1))

        return phst1, phst1_pred, at_pred

        # return (
        #     nn.functional.mse_loss(phst1_pred, phst1),
        #     nn.functional.mse_loss(at_pred, at_onehot),
        # )


class VisualAgentPPO(nn.Module):
    def __init__(
        self,
        input_shape,
        num_actions,
        num_hidden=512,
        device="cuda",
        smaller=False,
        recurrent=1024,
        curious=False,
    ):
        super(VisualAgentPPO, self).__init__()
        self.convs, num_fc = make_base_conv(input_shape, smaller)
        self.num_actions = num_actions

        if recurrent:
            self._recurrent = recurrent
            self.gru = nn.GRU(num_fc, recurrent)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)
            num_fc = recurrent

        self.curious = True
        if curious:
            self.icm = ICM(input_shape, num_actions, smaller)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

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
        self, x,
        hxs):  # hxs is size [Nbatch, 512], must be 0 at start of episode
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

    def ppo_update(
        self,
        ppo_epochs,
        mini_batch_size,
        states,
        next_states,
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

        final_forward_loss = None if not self.curious else 0
        final_inverse_loss = None if not self.curious else 0

        for _ in tqdm(range(ppo_epochs)):
            for (
                    state,
                    next_state,
                    action,
                    old_log_probs,
                    return_,
                    advantage,
                    r_state,
            ) in ppo_iter(
                    mini_batch_size,
                    states,
                    next_states,
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
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) *
                    advantage)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                optimizer.zero_grad()
                loss = 0.5 * critic_loss + actor_loss - 0.05 * entropy
                if self.curious:
                    ns, = next_state
                    phst1, phst1_pred, at_pred = model.icm(state,
                                                           ns, action)
                    at_onehot = nn.functional.one_hot(action.squeeze(),
                                                      self.num_actions).float()
                    forward_loss = nn.functional.mse_loss(phst1_pred, phst1)
                    inverse_loss = nn.functional.binary_cross_entropy(at_pred, at_onehot)
                    # loss += 10.0*(0.8 * inverse_loss + 0.2 * forward_loss)
                    loss += 10.0*forward_loss

                loss.backward()
                optimizer.step()

                final_loss += loss.detach().item()
                factor_loss += actor_loss.detach().item()
                fcritic_loss += critic_loss.detach().item()
                fentropy_loss += entropy.detach().item()
                if self.curious:
                    final_forward_loss += forward_loss.detach().item()
                    final_inverse_loss += inverse_loss.detach().item()
                final_loss_steps += 1

        if self.curious:
            final_inverse_loss /= final_loss_steps
            final_forward_loss /= final_loss_steps
        return (
            final_loss / final_loss_steps,
            factor_loss / final_loss_steps,
            fcritic_loss / final_loss_steps,
            fentropy_loss / final_loss_steps,
            final_forward_loss,
            final_inverse_loss,
        )

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def _t(l):
    return torch.cat([torch.FloatTensor(i) for i in l])


def main(device="cuda", env_name="snake", test=False, checkpoint_path=None):
    assert env_name in ["snake", "doom_basic", "doom_corridor", "doom_way"]
    recurrent = True
    curious = False
    recurrent_size = 1024 if recurrent else 0
    if not test:
        wandb.init(project="snake-pytorch-ppo", tags=env_name)
    num_envs = 64
    num_viz_train = 2
    if test:
        num_envs = 4
        num_viz_train = 4
    num_steps = 32
    if env_name == "snake":
        env_fac = lambda: gym.make(
            "snakenv-v0", gs=20, main_gs=22, num_fruits=1)
    elif env_name == "doom_basic":
        env_fac = lambda: gym.make("VizdoomBasic-v0")
    elif env_name == "doom_corridor":
        env_fac = lambda: gym.make("VizdoomCorridor-v0")
    elif env_name == "doom_way":
        curious = True
        env_fac = lambda: gym.make("VizdoomMyWayHome-v0")

    reward_mult = 1.0 if env_name in ["snake", "doom_way"] else 0.01
    if env_name is "doom_corridor":
        reward_mult = 0.1
    skip = 1 if env_name not in ["doom_corridor", "doom_way"] else 2

    if env_name == "doom_way":
        skip = 4
        reward_mult = 1

    m = EnvManager(
        env_fac,
        num_envs,
        pytorch=True,
        num_viz_train=num_viz_train,
        reward_mult=reward_mult,
        skip=skip,
    )
    s = m.state.shape

    if env_name == "snake":
        model = VisualAgentPPO(
            (1, s[-1], s[-1]),
            4,
            device=device,
            recurrent=recurrent_size,
            smaller=True,
            curious=curious,
        ).to(device)
    elif env_name in ["doom_basic", "doom_way"]:
        model = VisualAgentPPO(
            (3, s[2], s[3]),
            3,
            device=device,
            recurrent=recurrent_size,
            smaller=True,
            curious=curious,
        ).to(device)
    elif env_name == "doom_corridor":
        model = VisualAgentPPO(
            (3, s[2], s[3]),
            7,
            device=device,
            recurrent=recurrent_size,
            smaller=True,
            curious=curious,
        ).to(device)

    if checkpoint_path is not None:
        model.load(checkpoint_path)

    idx = 0
    batch_num = 0
    episode_num = 0

    recurrent_state = torch.zeros((num_envs, recurrent_size))

    while True:
        states = []
        next_states = []
        r_states = []
        values = []
        rewards = []
        dones = []
        actions = []
        log_probs = []
        scores = []
        rewards_for_tracking = []

        with torch.no_grad():
            for i in range(num_steps):
                if recurrent:
                    r_states.append(recurrent_state.cpu())
                dist, v, recurrent_state = model(
                    torch.FloatTensor(m.state).to(device),
                    recurrent_state.to(device))
                idx += num_envs
                if not test:
                    acts = dist.sample()
                else:
                    acts = dist.logits.max(1).indices.view(-1)
                ost, nst, r, d, idicts = m.apply_actions(acts.tolist())

                if curious:
                    phst1, phst1_pred, _ = model.icm(torch.FloatTensor(ost).to(device), torch.FloatTensor(nst).to(device), acts.unsqueeze(1))

                if not test:
                    states.append(ost)
                    reward_addition = 0
                    if curious:
                        next_states.append(nst)
                        reward_addition = 0.5 * F.mse_loss(phst1_pred, phst1, reduce=False).sum(-1).unsqueeze(-1) * 0.01
                        reward_addition = reward_addition.cpu().numpy()

                    rewards.append(r + reward_addition)
                    rewards_for_tracking.append((r+reward_addition).mean())
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
                    scores.extend([
                        idict["score"] for idict in idicts if "score" in idict
                    ])

        if not test:
            gae_ = compute_gae(
                model(
                    torch.FloatTensor(m.state).to(device),
                    recurrent_state.to(device))[1].cpu(),
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
            if curious:
                next_states = _t(next_states)
                # vec_st1s = _t(vec_st1s)
                # forwards = _t(forwards)
                # inverses= _t(inverses)

            curious_data = None if not curious else (next_states,)

            (
                loss,
                actor_loss,
                critic_loss,
                entropy_loss,
                forward_loss,
                inverse_loss,
            ) = model.ppo_update(
                2,
                min(num_envs*num_steps, 512),
                states,
                curious_data,
                actions,
                log_probs,
                gae,
                advantage,
                r_states,
            )
            batch_num += 1
            score = 0 if not scores else max(scores)
            log_dict = {
                "loss": loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "score": score,
                "steps": idx,
                "episodes": episode_num,
                "avg_reward": np.mean(rewards_for_tracking)
            }
            rewards_for_tracking = []
            if model.curious:
                log_dict.update({
                    "forward_loss": forward_loss,
                    "inverse_loss": inverse_loss,
                })
            wandb.log(
                log_dict,
                step=batch_num,
            )

        if os.path.exists("/tmp/debug_jari"):
            try:
                os.remove("/tmp/debug_jari")
            except Exception:
                pass
            import ipdb

            ipdb.set_trace()


if __name__ == "__main__":
    # main()
    argh.dispatch_command(main)
