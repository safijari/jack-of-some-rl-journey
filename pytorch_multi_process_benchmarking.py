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
    return gym.make("VizdoomDeathmatch-v0")

def _main(input_queue: Queue, output_queue: Queue, num_envs=32):
    print('spawned')
    reward_mult = 0.1
    skip = 4

    m = EnvManager(make_doom_deathmatch, num_envs, pytorch=True, num_viz_train=0, reward_mult=reward_mult, skip=skip)
    s = m.state

    output_queue.put(s)

    print('ready')
    while True:
        actions = input_queue.get()
        _, r, d, idicts = m.apply_actions(actions)
        output_queue.put((m.state, r, d, idicts))

def get_from_queues_synchronous(procs_queues):
    row = []
    for _, _, outq in procs_queues:
        row.append(outq.get())
    return row

def tcat(l, idx):
    return torch.cat([torch.Tensor(i[idx]) for i in l])

def main(num_procs=10, num_envs=32, num_steps=32):
    wandb.init(project="snake-pytorch-ppo", tags='deathmatch_parallel')
    idx = 0
    batch_num = 0
    device = 'cuda'
    recurrent = True
    recurrent_size = 512 if recurrent else 0
    recurrent_state = torch.zeros((num_envs*num_procs, recurrent_size))

    obs_shape = (3, 240, 320)
    num_actions = 7

    model = VisualAgentPPO((3, 240, 320), num_actions, device=device, recurrent=recurrent_size, smaller=True).to(device)
    storage = RolloutStorage(num_steps, num_envs*num_procs, obs_shape, num_actions, recurrent_size)


    procs_queues = []
    for i in tqdm(range(num_procs)):
        inq, outq = Queue(), Queue()
        p = mp.Process(target=_main, args=(inq, outq, num_envs))
        p.daemon = True
        p.start()
        procs_queues.append((p, inq, outq))

    row = get_from_queues_synchronous(procs_queues)

    tq = None
    while True:
        # outs = []
        for i in range(num_steps):
            s = storage.step
            with torch.no_grad():
                act_dist, vals, recurrent_state = model(storage.obs[s].to(device), storage.recurrent_states[s].to(device))

            action_sample = act_dist.sample() # num_envs * num_procs

            row = []
            for _i, (_, inq, _) in enumerate(procs_queues):
                inq.put(action_sample[_i*num_envs:(_i+1)*num_envs].cpu())

            for _, _, outq in procs_queues:
                row.append(outq.get())  # "next state, curr reward, curr done"

            state, rewards, dones = tcat(row, 0), tcat(row, 1), tcat(row, 2)
            for i, d in enumerate(dones):
                if d:
                    recurrent_state[i] = 0
            storage.insert(state, recurrent_state, action_sample.unsqueeze(1), act_dist.log_prob(action_sample).unsqueeze(1), vals, rewards, 1-dones)

            if tq is None:
                tq = tqdm()
            tq.update(num_procs*num_envs)
            idx += num_procs*num_envs

        with torch.no_grad():
            _, next_vals, _ = model(state.to(device), recurrent_state.to(device))

        storage.compute_returns(next_vals)

        _, actor_loss, critic_loss, entropy_loss = model.ppo_update_generator(storage.generate, 512 + 256 + 128, 4, 0.1)

        storage.after_update()

        wandb.log(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "steps": idx,
            },
            step=batch_num,
        )

        if batch_num % 10 == 0:
            model.save(f'/home/jack/rl_weights/deathmatch_parallel_{batch_num}.pth')

        batch_num += 1

        if os.path.exists("/tmp/debug_jari"):
            try:
                os.remove("/tmp/debug_jari")
            except Exception:
                pass
            import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # main()
    argh.dispatch_command(main)
