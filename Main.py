import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
import time as T
from matplotlib import pyplot as plt
from TwinDelayedDDPG.TD3 import TD3
from TwinDelayedDDPG.Replay_Buffer  import Replay_Buffer
import torch
import os
import json

NumberOfPS = SimulationParams.NumberOfPS

def evaluate_policy(Env, policy, eval_episodes = 10):
  avg_reward = dict()
  action = dict()
  states = dict()
  for i in range(1,NumberOfPS+1):
     avg_reward[i] = 0 
  for _ in range(eval_episodes):
    time = 0
    for i in range(1,NumberOfPS+1):
      states[i]= Env.reset(ps = i)
    while time < 200:
      for i in range(1,NumberOfPS+1):
        action[i] = policy[i].select_action(states[i])
        action[i] = abs(action[i])
      # print(states[1])
      states, rewards, dones, terminal = Env.step(action= action, time= time)
      states = {
          i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
      }
      time += 1
      for i in range(1,NumberOfPS+1):
        avg_reward[i] += rewards[i]
    for i in range(1,NumberOfPS+1):
        avg_reward[i] = avg_reward[i]/eval_episodes
  return avg_reward
save_models = True
enter = False
enter2 = False
expl_val = 1
expl_noise_min =5 
epsilon = 5e-3
expl_noise = 100
eval_freq = 1.5e3
batch_size = 100
tau = 0.005
discount = 0.99
noise_clip = 100
policy_noise = 100
policy_freq = 5
max_timesteps = 2e2
start_timesteps = 2e3
total_timesteps = 0
episode_num = dict()
episode_reward = dict()
episode_timesteps = dict()
timesteps_since_eval = dict()
done = dict()
done2 = dict()
is_completely_done = dict()
rewards = dict()
rewards2 = dict()
terminal_rewards = dict()
terminal = dict()
terminal_rewards2 = dict()
terminal2 = dict()
states = dict()
actions = dict()
next_states = dict()
states2 = dict()
actions2 = dict()
next_states2 = dict()
Env = CustumEnv(NumberOfPS)
Env2 = CustumEnv(NumberOfPS)
model = 5
policy = dict()
replay_buffer = dict()
Evaluations = dict()
for i in range(1,NumberOfPS+1):
  states[i] = Env.reset(i)
  states2[i] = Env2.reset(i)

  policy[i] = TD3(state_dim= SimulationParams.NumberOfTCh + 2, epsilon= epsilon, action_dim= 1, max_action= 500)
  replay_buffer[i] = Replay_Buffer()
  Evaluations[i] = []


policy[1].load(f"{1}th PS", f"./pytorch_models/Model{model}")
policy[2].load(f"{2}th PS", f"./pytorch_models/Model{model}")

AoI_dict = {"1":[] , "2":[]}
Power_dict = {"1":[] , "2":[]}
AoI_dict2 = {"1":[] , "2":[]}
Power_dict2 = {"1":[] , "2":[]}
Bits_Dict = {"1":[] , "2":[]}
while total_timesteps < max_timesteps:
  for ps in range(1,NumberOfPS+1):
    actions[ps]  = policy[ps].select_action(states[ps])
    actions[ps] = abs(actions[ps])
  print(f"Time: {total_timesteps}")
  print(f"States:{states}")

  # actions2 = {1:np.array([np.random.randint(400,500)]), 2:np.array([np.random.randint(400,500)])}
  for ps in range(1,NumberOfPS+1):
    if states2[ps][2] <= 100:
      actions2[ps] = np.array([0])
    else:
      if ps == 1:
        #  actions2[ps] = np.array([np.random.randint(60,70)])
         actions2[ps] = np.array([30])
      else:
         actions2[ps] = np.array([12])
  print(f"Action: {actions}, State: {states}, reward: {rewards}")
  next_states2, rewards2, done2, terminal2 = Env2.step(actions2, total_timesteps)    
  next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
  print(f"Next state: {next_states}")
  # T.sleep(5)
  next_states2 = {
          i: np.array(list(next_states2[0][f"ps{i}"]) + [next_states2[1][f"ps{i}"]] + [next_states2[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
  }
  next_states = {
          i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
  }
  if next_states[ps][2] == 0 and enter == False:
   enter = True
  if next_states2[ps][2] == 0 and enter2 == False:
   enter2 = True
  # print(f"Next-state: {next_states}")
  for ps in range(1,NumberOfPS+1):
   if enter == True:
    Power_dict[f"{ps}"].append(actions[ps])
    AoI_dict[f"{ps}"].append(states[ps][1])
    Bits_Dict[f"{ps}"].append(states[ps][2])
   if enter2 == True:
    Power_dict2[f"{ps}"].append(actions2[ps])
    AoI_dict2[f"{ps}"].append(states2[ps][1])
  # print(f"Sate: {states[1]}")
 
  # print(f"reward: {rewards[1]}")
  # print(f"Next state: {next_states[1]}")
  # T.sleep(2)
  states = next_states
  states2 = next_states2

  total_timesteps += 1
print(f"Average power usage for ps1 with RL scheduler: {sum(Power_dict['1'])/max_timesteps}")
print(f"Average power usage for ps2 with RL scheduler: {sum(Power_dict['2'])/max_timesteps}")
# print(f"Average power usage for ps1 without RL scheduler: {sum(Power_dict2['1'])/max_timesteps}")
# print(f"Average power usage for ps2 without RL scheduler: {sum(Power_dict2['2'])/max_timesteps}")
print(f"Average AoI for ps1 with RL scheduler: {sum(AoI_dict['1'])/max_timesteps}")
print(f"Average AoI for ps2 with RL scheduler: {sum(AoI_dict['2'])/max_timesteps}")
# print(f"Average AoI for ps1 without RL scheduler: {sum(AoI_dict2['1'])/max_timesteps}")
# print(f"Average AoI for ps2 without RL scheduler: {sum(AoI_dict2['2'])/max_timesteps}")


plt.figure(1)
plt.title("AoI change with RL power scheduler")
# plt.ylim(0,50)
plt.plot(AoI_dict["1"] , color = "green", label = "AoI ps1")
plt.plot(AoI_dict["2"] , color = "blue", label = "AoI ps2")
plt.axhline(y = 25, color = "red", linestyle = 'solid', label = "deadline ps2", )
plt.axhline(y = 15, color = "orange", linestyle = 'solid', label = "deadline ps1", )
plt.legend()

plt.figure(2)
plt.title("Power change with RL power scheduler")
plt.plot(Power_dict["1"],  color = "green", label = "AoI ps1")
plt.plot(Power_dict["2"],  color = "blue", label = "AoI ps2")
plt.legend()

# plt.figure(3)
# plt.title("AoI change without RL power scheduler")
# plt.plot(AoI_dict2["1"] , color = "green", label = "AoI ps1")
# plt.plot(AoI_dict2["2"] , color = "blue", label = "AoI ps2")
# plt.axhline(y = 20, color = "red", linestyle = 'solid', label = "deadline ps2", )
# plt.axhline(y = 15, color = "orange", linestyle = 'solid', label = "deadline ps1", )
# plt.legend()

# plt.figure(4)
# plt.title("Power change without RL power scheduler")
# plt.plot(Power_dict2["1"],  color = "green", label = "AoI ps1")
# plt.plot(Power_dict2["2"],  color = "blue", label = "AoI ps2")
# plt.legend()

plt.figure(5)
plt.title("Bits with RL power scheduler")
plt.plot(Bits_Dict["1"],  color = "green", label = "AoI ps1")
plt.plot(Bits_Dict["2"],  color = "blue", label = "AoI ps2")
plt.legend()
plt.show()

# if not os.path.exists("./results"):
#     os.makedirs("./results")
# if save_models and not os.path.exists("./pytorch_models"):
#     os.makedirs("./pytorch_models")

# for ps in range(1,NumberOfPS+1):
#   terminal_rewards[ps] = 0
#   done[ps] = True
#   episode_num[ps] = 0
#   episode_reward[ps] = 0
#   episode_timesteps[ps] = 0
#   timesteps_since_eval[ps] = 0

# while total_timesteps < max_timesteps:
#   for ps in range(1,NumberOfPS+1):
#     if episode_timesteps[ps] >= 300:
#       if total_timesteps != 0:
#         print(f"PS: {ps}, Total Timesteps: {total_timesteps}, Episode Num: {episode_num[ps]}, Reward: {episode_reward[ps]}")
#         policy[ps].train(replay_buffer[ps], episode_timesteps[ps], batch_size, discount, tau, noise_clip, policy_freq)
#       if timesteps_since_eval[ps] >= eval_freq:
#         timesteps_since_eval[ps] %= eval_freq
#         Evaluations[ps].append(evaluate_policy(Env= Env, policy= policy))
#         policy[ps].save(f"{ps}th PS", "./pytorch_models")
#         np.save(f"./results/{ps}th PS", Evaluations[ps])
#         with open(f"./results/{ps}th PS.txt", "w") as fp:
#           json.dump(Evaluations[ps], fp)

#       states[ps] = Env.reset(ps= ps)
#       done[ps] = False
#       episode_reward[ps] = 0
#       episode_timesteps[ps] = 0
#       episode_num[ps] += 1
#     if total_timesteps < start_timesteps:
#       actions[ps] = np.random.uniform(0,policy[ps].max_action,1)

#     else:
#         actions[ps] = policy[ps].select_action(states[ps])
#         actions[ps] = abs(actions[ps])
#         print(actions[ps])

#         # if np.random.random() < expl_val and actions[ps] != 0:
#         actions[ps] = (actions[ps] + np.random.normal(0,expl_noise,1)).clip(-policy[ps].max_action, policy[ps].max_action) 
#         actions[ps] = abs(actions[ps])
#   # print(states)
#   # print(actions)
#   next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
#   # print(next_states)
#   next_states = {
#           i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
#           for i in range(1, NumberOfPS + 1)
#   }
#   for ps in range(1,NumberOfPS+1):
#     episode_reward[ps] += rewards[ps]
#     if done[ps] == 1:
#       terminal_rewards[ps] += rewards[ps]
#       if terminal[ps] == 1 or episode_timesteps[ps] == 300:
#         replay_buffer[ps].add((states[ps], actions[ps], next_states[ps], terminal_rewards[ps], done[ps]))
#         terminal_rewards[ps] = 0
#     else:
#       replay_buffer[ps].add((states[ps], actions[ps],next_states[ps], rewards[ps], done[ps]))
#     episode_timesteps[ps] += 1
#     timesteps_since_eval[ps] += 1

#   states = next_states
#   total_timesteps += 1
#   # expl_val = expl_val - epsilon if expl_val > expl_noise_min else expl_noise_min 
#   expl_noise = expl_noise - epsilon if expl_noise > expl_noise_min else expl_noise_min
#   print(expl_noise)
# for ps in range(1,NumberOfPS+1):
#   Evaluations[ps].append(evaluate_policy(Env= Env, policy= policy))
#   if save_models == True:
#     policy[ps].save(f"{ps}th PS", "./pytorch_models")
#     np.save(f"./results/{ps}th PS", Evaluations[ps])
