import os
import json
import argparse
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from net import *
from env import *
# from oldenv import *
from a2c import *


# DEFAULT_STIMULI_LOGLR = np.array(
#     [-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9],
#     dtype=np.float64,
# )

DEFAULT_STIMULI_LOGLR = np.array(
    [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    dtype=np.float64,
)


def parse_stimuli_loglr(raw_values):
    cleaned = raw_values.strip()
    if cleaned.startswith('[') and cleaned.endswith(']'):
        cleaned = cleaned[1:-1]

    if not cleaned:
        raise ValueError('stimuli_loglr must contain at least one value')

    values = [float(item.strip()) for item in cleaned.split(',') if item.strip()]
    return np.array(values, dtype=np.float64)


# note: vectorizing wrapper only works under this protection
if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = '20260616_reward=1.0_sample_cost=0.01_urgency_cost=0.0_logLR=[-0.9,-0.7,-0.5,-0.3,0.3,0.5,0.7,0.9]_max_samples=10000_max_steps=10000_epNum=1000000', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 64, help = 'lstm hidden size')

    # environment parameters
    parser.add_argument('--num_trials', type = int, default = 1, help = 'number of trials per episode')
    parser.add_argument('--max_samples', type = int, default = 10000, help = 'number of samples per trial')
    parser.add_argument('--max_steps', type = int, default = 10000, help = 'maximum steps per trial')
    parser.add_argument('--reward', type = float, default = 1.0, help = 'reward for correct answer')
    parser.add_argument('--sampling_cost', type = float, default = 0.01, help = 'sampling cost')
    parser.add_argument('--urgency_cost', type = float, default = 0.00, help = 'urgency cost')
    parser.add_argument('--num_stimuli', type = int, default = 16, help = 'number of stimuli')
    parser.add_argument(
        '--stimuli_loglr',
        type = str,
        default = ','.join(str(value) for value in DEFAULT_STIMULI_LOGLR),
        help = 'comma-separated stimulus logLR values',
    )

    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 1500000, help = 'training episodes')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch_size')
    parser.add_argument('--gamma', type = float, default = 1.0, help = 'temporal discount')
    parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
    parser.add_argument('--beta_v', type = float, default = 0.05, help = 'value loss coefficient')
    parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')

    args = parser.parse_args()
    stimuli_loglr = parse_stimuli_loglr(args.stimuli_loglr)

    if len(stimuli_loglr) != args.num_stimuli:
        raise ValueError(
            f'num_stimuli ({args.num_stimuli}) must match the number of logLR values ({len(stimuli_loglr)}).'
        )

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    config = {
        'jobid': args.jobid,
        'hidden_size': args.hidden_size,
        'num_trials': args.num_trials,
        'max_samples': args.max_samples,
        'max_steps': args.max_steps,
        'reward': args.reward,
        'sampling_cost': args.sampling_cost,
        'urgency_cost': args.urgency_cost,
        'num_stimuli': args.num_stimuli,
        'stimuli_logLR': stimuli_loglr.tolist(),
        'num_episodes': args.num_episodes,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'gamma': args.gamma,
        'lamda': args.lamda,
        'beta_v': args.beta_v,
        'beta_e': args.beta_e,
        'max_grad_norm': args.max_grad_norm,
    }
    with open(os.path.join(exp_path, 'training_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    
    num_batches = args.num_episodes // args.batch_size
    lr_schedule = two_phase_linear(
        total = num_batches,
        start = args.lr,
        final = 1e-5,
    )
    entropy_schedule = two_phase_linear(
        total = num_batches,
        start = args.beta_e,
        final = 0,
    )

    env = gym.vector.SyncVectorEnv([
        lambda seed=seeds[i]: 
        # MetaLearningWrapper(
            SequentialInferenceEnv(
                num_trials = args.num_trials,
                max_samples = args.max_samples,
                num_stimuli = args.num_stimuli,
                reward = args.reward,
                sampling_cost = args.sampling_cost,
                urgency_cost = args.urgency_cost,
                stimuli_logLR = stimuli_loglr,
                max_steps = args.max_steps,
                seed = seed,
            )
            # )
        for i in range(args.batch_size)
    ])

    # set network
    net = SharedGRURecurrentActorCriticPolicy(
        feature_dim = env.single_observation_space.shape[0],
        action_dim = env.single_action_space.n,
        gru_hidden_dim = args.hidden_size,
    )

    # set model
    model = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        max_grad_norm = args.max_grad_norm,
        lr_schedule = lr_schedule,
        entropy_schedule = entropy_schedule,
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 10
    )
    
    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))

    N = len(data['loss'])  # or len(data['reward']) if same length
    if N == 0:
        raise RuntimeError('Training completed without collecting any loss values.')

    num_segments = min(100, N)
    segment_edges = np.linspace(0, N, num_segments + 1, dtype=int)

    # average the episode reward and loss
    avg_episode_reward = []
    avg_loss = []
    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        avg_episode_reward.append(np.mean(data['episode_reward'][start:end]))
        avg_loss.append(np.mean(data['loss'][start:end]))

    episode_percentage = np.linspace(0, 100, num_segments)

    # plot the episode reward
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(episode_percentage, avg_episode_reward)
    plt.xlabel('Episodes Percentage')
    plt.ylabel('Average Episode Reward')
    plt.title('Training Reward (Averaged)')
    plt.savefig(os.path.join(exp_path, f'training_reward.png'))
    plt.close()

    # plot the loss
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(episode_percentage, avg_loss)
    plt.xlabel('Episodes Percentage')
    plt.ylabel('Average Loss')
    plt.title('Training Loss (Averaged)')
    plt.savefig(os.path.join(exp_path, f'training_loss.png'))
    plt.close()
