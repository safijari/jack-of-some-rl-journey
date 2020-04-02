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
from pytorch_common import _t, VisualAgentPPO, CuriosityTracker


def main(device="cuda", env_name="snake", test=False, checkpoint_path=None):
    assert env_name in ['snake', 'doom_basic', 'doom_corridor', 'doom_way', 'doom_deathmatch', 'doom_line']
    recurrent = True
    recurrent_size = 256 if recurrent else 0
    if not test:
        wandb.init(project="snake-pytorch-ppo", tags=env_name)
    num_envs = 16*4
    num_viz_train = 4
    if test:
        num_envs = 4
        num_viz_train = 4
    num_steps = 8*2
    if env_name == "snake":
        env_fac = lambda: gym.make("snakenv-v0", gs=20, main_gs=22, num_fruits=1)
    elif env_name == "doom_basic":
        env_fac = lambda: gym.make("VizdoomBasic-v0")
    elif env_name == "doom_corridor":
        env_fac = lambda: gym.make("VizdoomCorridor-v0")
    elif env_name == "doom_way":
        env_fac = lambda: gym.make("VizdoomMyWayHome-v0")
    elif env_name == "doom_deathmatch":
        env_fac = lambda: gym.make("VizdoomDeathmatch-v0")
    elif env_name == "doom_line":
        env_fac = lambda: gym.make("VizdoomDefendLine-v0")

    reward_mult = 1.0 if env_name in ['snake', 'doom_way', 'doom_deathmatch', 'doom_line'] else 0.01
    if env_name is 'doom_corridor':
        reward_mult = 0.1
    skip = 1 if env_name not in ['doom_corridor', 'doom_way', 'doom_deathmatch'] else 4
    skip = 4

    m = EnvManager(env_fac, num_envs, pytorch=True, num_viz_train=num_viz_train, reward_mult=reward_mult, skip=skip)
    s = m.state.shape

    if env_name == "snake":
        model = VisualAgentPPO((1, s[-1], s[-1]), 4, device=device, recurrent=recurrent_size, smaller=True).to(device)
    elif env_name in ["doom_basic", "doom_way"]:
        model = VisualAgentPPO((3, s[2], s[3]), 3, device=device, recurrent=recurrent_size, smaller=True).to(device)
    elif env_name == "doom_corridor":
        model = VisualAgentPPO((3, s[2], s[3]), 7, device=device, recurrent=recurrent_size, smaller=True).to(device)
    elif env_name == "doom_deathmatch":
        model = VisualAgentPPO((3, s[2], s[3]), 7, device=device, recurrent=recurrent_size, smaller=True).to(device)
    elif env_name == "doom_line":
        model = VisualAgentPPO((3, s[2], s[3]), 3, device=device, recurrent=recurrent_size, smaller=True).to(device)

    curiosity_target = CuriosityTracker((3, s[2], s[3])).to(device)
    curiosity_model = CuriosityTracker((3, s[2], s[3])).to(device)
    for p in curiosity_target.parameters():
        p.requires_grad = False

    if checkpoint_path is not None:
        model.load(checkpoint_path)

    idx = 0
    batch_num = 0
    episode_num = 0

    recurrent_state = torch.zeros((num_envs, recurrent_size))

    while True:
        states = []
        state_next_state = []
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
                if not test:
                    acts = dist.sample()
                else:
                    acts = dist.logits.max(1).indices.view(-1)
                ost, r, d, idicts = m.apply_actions(acts.tolist())

                curiosity_output = curiosity_model(torch.FloatTensor(ost).to(device))
                curiosity_target_output = curiosity_target(torch.FloatTensor(ost).to(device))

                intrinsic_reward = (curiosity_output - curiosity_target_output).mean(1).unsqueeze(1).cpu().numpy()

                if not test:
                    for i, (st, dn) in enumerate(zip(ost, d)):
                        if not dn:
                            state_next_state.append((st, m.state[i]))

                    states.append(ost)
                    rewards.append(r + intrinsic_reward)
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

        if not test:
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

            loss, actor_loss, critic_loss, entropy_loss = model.ppo_update(
                4, min(num_envs*num_steps, 1024), states, actions, log_probs, gae, advantage, r_states
            )

            curiosity_model.optimizer.zero_grad()
            intrinsic_target = curiosity_target(states.to(device))
            intrinsic_actual = curiosity_model(states.to(device))

            intrinsic_loss = nn.functional.mse_loss(intrinsic_actual, intrinsic_target)
            intrinsic_loss.backward()
            curiosity_model.optimizer.step()

            batch_num += 1
            score = 0 if not scores else max(scores)
            wandb.log(
                {
                    "loss": loss,
                    "intrinsic_loss": intrinsic_loss.cpu().detach().item(),
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy_loss": entropy_loss,
                    "steps": idx,
                    "episodes": episode_num
                },
                step=batch_num,
            )

        if os.path.exists("/tmp/debug_jari"):
            try:
                os.remove("/tmp/debug_jari")
            except Exception:
                pass
            import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # main()
    argh.dispatch_command(main)
