import torch
import numpy as np 
import pandas as pd
import gymnasium as gym
import os

from env import *
from a2c import BatchMaskA2C
from net import *
from replaybuffer import *


if __name__ == '__main__':

    data = []

    env = gym.vector.SyncVectorEnv([
        lambda: 
        # MetaLearningWrapper(
            SequentialInferenceEnv(
            num_trials=1,
            max_samples=10000,
            max_steps=10000,
            reward=1.0,
            sampling_cost=0.01,
            urgency_cost=0.00,
            num_stimuli=16,
            stimuli_logLR=np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            seed=20250916,
            sequences=None,
            sequence_answers=None,
            )
        # )
        for i in range(1)
    ])

    path = os.getcwd() + '/train_taskoptrnns/results/Cond1/exp_20260616_reward=1.0_sample_cost=0.01_urgency_cost=0.0_logLR=[-0.9,-0.7,-0.5,-0.3,0.3,0.5,0.7,0.9]_max_samples=10000_max_steps=10000_epNum=1000000/'
    net = torch.load(path + 'net.pth', weights_only=False)
    net.eval()  # Set network to evaluation mode

    num_episodes = 50000
    batch_size = 1

    for episode in range(num_episodes):

        if episode % 1000 == 0:
            print(f'Simulating episode {episode}...')
        
        buffer = BatchReplayBuffer()
            
        dones = np.zeros(batch_size, dtype = bool)
        mask = torch.ones(batch_size)
        states_hidden = None

        obs, info = env.reset()
        obs = torch.Tensor(obs)
        action_mask = torch.tensor(np.stack(info['mask']))

        episode_data = []

        episode_data.append({
            'episode': episode,
            'correct_answer': info['correct_answer'][0],
            'stimuli_so_far': info['stimuli_so_far'][0],
            'current_stimulus': info['current_stimulus'][0],
            'sequence': info['sequence'][0],
            'test_idx': info['test_idx'][0],
        })

        while not all(dones):

            action, policy, log_prob, entropy, value, states_hidden = net(
                obs, states_hidden, action_mask,
            )
            value = value.view(-1)
            obs, reward, done, truncated, info = env.step(action)
            obs = torch.Tensor(obs)
            reward = torch.Tensor(reward)
            action_mask = torch.tensor(np.stack(info['mask']))

            episode_data[-1]['action'] = action.item()
            episode_data[-1]['reward'] = round(reward.item(), 4)
            episode_data[-1]['policy'] = policy.tolist()[0]
            episode_data[-1]['hidden_state'] = states_hidden.detach().cpu().numpy().tolist()[0]

            if done == False:
                episode_data.append({
                    'episode': episode,
                    'correct_answer': info['correct_answer'][0],
                    'stimuli_so_far': info['stimuli_so_far'][0],
                    'current_stimulus': info['current_stimulus'][0].tolist(),
                    'sequence': info['sequence'][0],
                    'test_idx': info['test_idx'][0],
                })

            dones = np.logical_or(dones, done)
            mask = (1 - torch.Tensor(dones))

        data.extend(episode_data)

    df = pd.DataFrame(data)
    df.to_json(path + "data.json", orient="records", lines=True)
    print("done")