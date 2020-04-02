import ray

ray.init(num_cpus=8)

import time
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
from common import RolloutStorage
from pytorch_common import _t, VisualAgentPPO
import torch.multiprocessing as mp
from multiprocessing import Queue


def make_doom_deathmatch():
    import vizdoomgym

    return gym.make("VizdoomCorridor-v0")


@ray.remote(num_cpus=0.5)
class Runner:
    def __init__(self, num_envs=32, reward_mult=0.001, skip=1):
        print("making envs")
        self.m = EnvManager(
            make_doom_deathmatch,
            num_envs,
            pytorch=True,
            num_viz_train=0,
            reward_mult=reward_mult,
            skip=skip,
        )
        print("done making envs")

    def step(self, actions):
        _, r, d, idicts = self.m.apply_actions(actions)
        state_buffer = self.m.state
        rewards_buffer = r
        dones_buffer = d
        return state_buffer, rewards_buffer, dones_buffer, idicts


def tcat(l, idx):
    return torch.cat([torch.Tensor(i[idx]) for i in l])
    # return torch.cat([i[idx] for i in l])


def main(num_procs=8, num_envs=32, num_steps=32):
    wandb.init(project="snake-pytorch-ppo", tags="deathmatch_parallel")
    idx = 0
    batch_num = 0
    device = "cuda"
    recurrent = True
    recurrent_size = 256 if recurrent else 0
    recurrent_state = torch.zeros((num_envs * num_procs, recurrent_size))

    obs_shape = (3, 240 // 2, 320 // 2)
    num_actions = 7

    model = VisualAgentPPO(
        obs_shape, num_actions, device=device, recurrent=recurrent_size, smaller=True
    ).to(device)
    storage = RolloutStorage(
        num_steps, num_envs * num_procs, obs_shape, num_actions, recurrent_size
    )

    runners = [Runner.remote(num_envs) for _ in range(num_procs)]

    tq = None
    once_done = False
    while True:
        scores = []
        for i in range(num_steps):
            s = storage.step

            with torch.no_grad():
                act_dist, vals, recurrent_state = model(
                    storage.obs[s].to(device), storage.recurrent_states[s].to(device)
                )
            action_sample = act_dist.sample()  # num_envs * num_procs

            row = ray.get(
                [
                    r.step.remote(
                        action_sample[_i * num_envs : (_i + 1) * num_envs].cpu()
                    )
                    for _i, r in enumerate(runners)
                ]
            )

            state, rewards, dones = tcat(row, 0), tcat(row, 1), tcat(row, 2)
            for i, d in enumerate(dones):
                if d:
                    for r in row:
                        for _d in r[-1]:
                            if "score" in _d:
                                scores.append(_d["score"])
                    recurrent_state[i] = 0

            storage.insert(
                state,
                recurrent_state,
                action_sample.unsqueeze(1),
                act_dist.log_prob(action_sample).unsqueeze(1),
                vals,
                rewards,
                1 - dones,
            )
            # print('inserted state')

            if tq is None:
                tq = tqdm()
            tq.update(num_procs * num_envs)
            idx += num_procs * num_envs

            # print('end loop')
        with torch.no_grad():
            _, next_vals, _ = model(state.to(device), recurrent_state.to(device))

        storage.compute_returns(next_vals)

        _, actor_loss, critic_loss, entropy_loss = model.ppo_update_generator(
            storage.generate, 512 + 256 + 128, 2, 0.1
        )

        storage.after_update()

        if len(scores) == 0:
            scores = [0]

        wandb.log(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "steps": idx,
                "score": max(scores),
            },
            step=batch_num,
        )

        if batch_num % 10 == 0:
            model.save(f"/home/jack/rl_weights/deathmatch_parallel_{batch_num}.pth")

        batch_num += 1

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
