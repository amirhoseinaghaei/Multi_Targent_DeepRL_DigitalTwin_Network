import numpy as np 
import os 
import gym
import torch 
from TwinDelayedDDPG.TD3 import TD3
from TwinDelayedDDPG.Replay_Buffer import Replay_Buffer
env_name = "sss"
seed  = 0 
start_timesteps = 1e4
eval_freq = 5e3
max_timesteps = 1e6
save_models = True
expl_noise = 0.1
batch_size = 100
tau = 0.005
discount = 0.99
noise_clip = 0.5
policy_noise = 0.2
policy_freq = 2


def evaluate_policy(policy, eval_episodes = 10):
    avg_reward = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("--------------------------------------------")
    print(f"Average Reward iver the Evaluation step: {avg_reward}")
    print("---------------------------------------------")
    return avg_reward

file_name = f"TD3{env_name}{seed}"
print("-------------------------")
print(f"Settings: {file_name}")
print("-------------------------")




env = gym.make(env_name)
env.seed(seed)
torch.manual_seed(seed= seed)
np.random.seed(seed=seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim= state_dim, action_dim= action_dim, max_action= max_action)
replay_buffer = Replay_Buffer()
evaluations = [evaluate_policy(policy= policy)]
