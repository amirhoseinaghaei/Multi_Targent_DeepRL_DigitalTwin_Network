import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
import time as T
from matplotlib import pyplot as plt
from MDDPG.Replay_Buffer  import Replay_Buffer
from MDDPG.Central_Critic import Central_Critic
import torch
import os
import json




NumberOfPS = SimulationParams.NumberOfPS

def add_gaussian_noise(action, expl_noise =0.002, min_action=0, max_action=1.0):
    noise = np.random.normal(0, max_action * expl_noise, size= 1)
    noisy_action = action + noise
    clipped_action = np.clip(noisy_action, min_action, max_action)
    return clipped_action

def evaluate_policy(Env, policy, eval_episodes = 30):
  signal = False
  scores = dict()
  cumulative = dict()
  avg_reward = dict()
  AoIs = dict()
  powers = dict()
  cr_reward = dict()
  action = dict()
  states = dict()
  for i in range(1,NumberOfPS+1):
     avg_reward[i] = 0 
     AoIs[i] = 0
     powers[i] = 0
     cr_reward[i] = 0
     scores[i] = 0
  for j in range(eval_episodes):
    time = 0
    for i in range(1,NumberOfPS+1):
      cr_reward[i] = 0
      states[i]= Env.reset(ps = i)
      cumulative[i] = 0 

    while time < 200:
      for i in range(1,NumberOfPS+1):

        action[i] = policy[i].select_action(states[i])
        action[i] = abs(action[i])
      states, rewards, dones, terminal = Env.step(action= action, time= time)
      states = {
          i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
      }
      time += 1
      for i in range(1,NumberOfPS+1):
        avg_reward[i] += rewards[i]
#        AoIs[i] += states[i][1] * SimulationParams.deadlines[0]
        AoIs[i] += states[i][1]     
        powers[i] += action[i][0]
        cr_reward[i] += rewards[i]
    for i in range(1,NumberOfPS+1):
      scores[i] = scores[i] + 1 if cr_reward[i] > -100 else scores[i]
  for i in range(1,NumberOfPS+1):
      avg_reward[i] = avg_reward[i]/eval_episodes
      AoIs[i] = AoIs[i]/eval_episodes 
      powers[i] = powers[i]/eval_episodes
      scores[i] = scores[i]/eval_episodes
  signal = True
  for i in range(1,NumberOfPS+1):
    if avg_reward[i] < -300:
        signal = False
        break
  return avg_reward, signal , scores, AoIs , powers
save_models = True
expl_noise_min = 0.01
epsilon = 5e-4
eval_freq = 200
batch_size = 256
tau = 0.005
discount = 0.99
expl_noise = 0.1
eps = 0.5
delta_eps = 2e-5
eps_min = 0.02
noise_clip =  0.5
policy_freq = 2
max_timesteps = 50e3
start_timesteps = 20e3
total_timesteps = 0
episode_num = dict()
episode_reward = dict()
min_episode_reward = dict()
max_episode_reward = dict()
episode_timesteps = dict()
timesteps_since_eval = dict()
done = dict()
done2 = dict()
done3 = dict()
done4 = dict()
doneRandom = dict()
is_completely_done = dict()
rewards = dict()
rewards2 = dict()
rewards3 = dict()
rewards4 = dict()
rewardsRandom = dict()
terminal_rewards = dict()
terminal = dict()
terminal_rewards2 = dict()
terminal_rewards3 = dict()
terminal_rewards4 = dict()
terminal2 = dict()
terminal3 = dict()
terminal4 = dict()
terminalRandom = dict()
states = dict()
actions = dict()
next_states = dict()
states2 = dict()
actions2 = dict()
states3 = dict()
actions3 = dict()
states4 = dict()
actions4 = dict()
actionsRandom = dict()
statesRandom = dict()
next_states2 = dict()
next_states3 = dict()
next_states4 = dict()

next_statesRandom = dict()
Env = CustumEnv(NumberOfPS, 5)
Env2 = CustumEnv(NumberOfPS , 50)
Env3 = CustumEnv(NumberOfPS , 15)
Env4 = CustumEnv(NumberOfPS , 25)
EnvRandom = CustumEnv(NumberOfPS, 3)
Test = False
policy = dict()
policy2 = dict()
policy3 = dict()
policy4 = dict()
new_policy = dict()
policyRandom = dict()
Evaluations = []
Reward_Evaluations = []
replay_buffer = dict()

for i in range(1,NumberOfPS+1):
  states[i] = Env.reset(i)
  states2[i] = Env2.reset(i)
  states3[i] = Env3.reset(i)
  states4[i] = Env4.reset(i)

  statesRandom[i] = EnvRandom.reset(i)

  policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  policy2[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  policy3[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  policy4[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  replay_buffer[i] = Replay_Buffer()

central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, action_dim=  SimulationParams.NumberOfTCh)


if Test == True:

  policy[1].load(f"{1}th PS", f"./pytorch_models")
  policy[2].load(f"{2}th PS", f"./pytorch_models")
  policy[3].load(f"{3}th PS", f"./pytorch_models")
  policy[4].load(f"{4}th PS", f"./pytorch_models")
  policy[5].load(f"{5}th PS", f"./pytorch_models")
  # policy[6].load(f"{6}th PS", f"./pytorch_models/5ps_100ch_centralcritic_100bits_40w_80d_power_reward_powermonitor_paper")

  # policy2[1].load(f"{1}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy2[2].load(f"{2}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy2[3].load(f"{3}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy2[4].load(f"{4}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy2[5].load(f"{5}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy2[6].load(f"{6}th PS", f"./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper")
  # policy3[1].load(f"{1}th PS", f"./pytorch_models/5ps_15ch_centralcritic_100bits_40w_80d_power")
  # policy3[2].load(f"{2}th PS", f"./pytorch_models/5ps_15ch_centralcritic_100bits_40w_80d_power")
  # policy3[3].load(f"{3}th PS", f"./pytorch_models/5ps_15ch_centralcritic_100bits_40w_80d_power")
  # policy3[4].load(f"{4}th PS", f"./pytorch_models/5ps_15ch_centralcritic_100bits_40w_80d_power")
  # policy3[5].load(f"{5}th PS", f"./pytorch_models/5ps_15ch_centralcritic_100bits_40w_80d_power")

  # policy4[1].load(f"{1}th PS", f"./pytorch_models/5ps_25ch_centralcritic_100bits_40w_80d_power")
  # policy4[2].load(f"{2}th PS", f"./pytorch_models/5ps_25ch_centralcritic_100bits_40w_80d_power")
  # policy4[3].load(f"{3}th PS", f"./pytorch_models/5ps_25ch_centralcritic_100bits_40w_80d_power")
  # policy4[4].load(f"{4}th PS", f"./pytorch_models/5ps_25ch_centralcritic_100bits_40w_80d_power")
  # policy4[5].load(f"{5}th PS", f"./pytorch_models/5ps_25ch_centralcritic_100bits_40w_80d_power")


  # policy2[1].load(f"{1}th PS", f"./pytorch_models/Non_CooperativeWithPower5PS")
  # policy2[2].load(f"{2}th PS", f"./pytorch_models/Non_CooperativeWithPower5PS")
  # policy2[3].load(f"{3}th PS", f"./pytorch_models/Non_CooperativeWithPower5PS")
  # policy2[4].load(f"{4}th PS", f"./pytorch_models/Non_CooperativeWithPower5PS")
  # policy2[5].load(f"{5}th PS", f"./pytorch_models/Non_CooperativeWithPower5PS")
  # policy3[1].load(f"{1}th PS", f"./pytorch_models")
  # policy3[2].load(f"{2}th PS", f"./pytorch_models")
  # policy3[3].load(f"{3}th PS", f"./pytorch_models")
  # policy3[4].load(f"{4}th PS", f"./pytorch_models")
  # policy3[5].load(f"{5}th PS", f"./pytorch_models")

  Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0} 
  Max_Steps2 = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0} 
  Max_Steps3 = {1:0, 2:0, 3:0 , 4:0 ,5:0, 6:0} 
  Max_Steps4 = {1:0, 2:0, 3:0 , 4:0 ,5:0, 6:0} 
  Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0} 


  max_Steps = 0
  AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dict2 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict2 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dict3 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict3 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dict4 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict4 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_Dict2 = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_Dict3 = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_Dict4 = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}

  Bits_DictRandom = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}

  while total_timesteps < max_timesteps:
    for ps in range(1,NumberOfPS+1):
      actions[ps]  = policy[ps].select_action(states[ps])
      actions[ps] = abs(actions[ps])
      # actions2[ps]  = policy2[ps].select_action(states2[ps])
      # actions2[ps] = abs(actions2[ps])

      # # actions2[ps]  = policy2[ps].select_action(states2[ps])
      # # actions2[ps] = abs(actions2[ps])
      # actions3[ps]  = policy3[ps].select_action(states3[ps])
      # actions3[ps] = abs(actions3[ps])
      # actions4[ps]  = policy4[ps].select_action(states4[ps])
      # actions4[ps] = abs(actions4[ps])
      if statesRandom[ps][2] <= 100:
        actionsRandom[ps] = np.array([0])
      else:
          actionsRandom[ps] = np.array([np.random.uniform(0,1)])
    next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
    # next_states2, rewards2, done2, terminal2 = Env2.step(actions2, total_timesteps)    
    # next_states3, rewards3, done3, terminal3 = Env3.step(actions3, total_timesteps)    
    # next_states4, rewards4, done4, terminal4 = Env4.step(actions4, total_timesteps)    

    next_statesRandom, rewardsRandom, doneRandom, terminalRandom = EnvRandom.step(actionsRandom, total_timesteps)    
    # print("**************************************************************\n\n\n")
    # print(f"state: {states}")
    # print(f"action: {actions}")
    # print(f"Next state: {next_states}")
    # print("\n\n\n**************************************************************")

    # T.sleep(2)
    # next_states2 = {
    #         i: np.array(list(next_states2[0][f"ps{i}"]) + [next_states2[1][f"ps{i}"]] + [next_states2[2][f"ps{i}"]])
    #         for i in range(1, NumberOfPS + 1)
    # }
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
            
    }
    # next_states3 = {
    #         i: np.array(list(next_states3[0][f"ps{i}"]) + [next_states3[1][f"ps{i}"]] + [next_states3[2][f"ps{i}"]])
    #         for i in range(1, NumberOfPS + 1)
            
    # }
    # next_states4 = {
    #         i: np.array(list(next_states4[0][f"ps{i}"]) + [next_states4[1][f"ps{i}"]] + [next_states4[2][f"ps{i}"]])
    #         for i in range(1, NumberOfPS + 1)
            
    # }
    next_statesRandom = {
            i: np.array(list(next_statesRandom[0][f"ps{i}"]) + [next_statesRandom[1][f"ps{i}"]] + [next_statesRandom[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
            
    }
    # if terminal2[1] == False:

    for ps in range(1, NumberOfPS + 1):
      if next_states[ps][2] != 0 :
        Max_Steps[ps] += 1  
    
    # for ps in range(1, NumberOfPS + 1):
    #   if next_states2[ps][2] != 0 :
    #     Max_Steps2[ps] += 1  
    # for ps in range(1, NumberOfPS + 1):
    #   if next_states3[ps][2] != 0 :
    #     Max_Steps3[ps] += 1  
    for ps in range(1, NumberOfPS + 1):
      if next_statesRandom[ps][2] != 0 :
        Max_StepsRandom[ps] += 1 
    # if next_states[ps][2] == 0 :
    #   if enter == False:
    #     enter = True

    # if next_states2[ps][2] == 0 and enter2 == False:
    #  enter2 = True

    for ps in range(1,NumberOfPS+1):
      Power_dict[f"{ps}"].append(actions[ps])
      AoI_dict[f"{ps}"].append(states[ps][1])
      Bits_Dict[f"{ps}"].append(states[ps][2])
      # Power_dict2[f"{ps}"].append(actions2[ps])
      # AoI_dict2[f"{ps}"].append(states2[ps][1])
      # Bits_Dict2[f"{ps}"].append(states2[ps][2])
      # Power_dict3[f"{ps}"].append(actions3[ps])
      # AoI_dict3[f"{ps}"].append(states3[ps][1])
      # Bits_Dict3[f"{ps}"].append(states3[ps][2])
      # Power_dict4[f"{ps}"].append(actions4[ps])
      # AoI_dict4[f"{ps}"].append(states4[ps][1])
      # Bits_Dict4[f"{ps}"].append(states4[ps][2])
      Power_dictRandom[f"{ps}"].append(actionsRandom[ps])
      AoI_dictRandom[f"{ps}"].append(statesRandom[ps][1])
      Bits_DictRandom[f"{ps}"].append(statesRandom[ps][2])

  
    states = next_states
    # states2 = next_states2
    # states3 = next_states3
    # states4 = next_states4
    statesRandom = next_statesRandom

    total_timesteps += 1
  print(Max_Steps)
  print(f"Average power usage for ps1 with RL scheduler - Cooperative: {sum(Power_dict['1'])/len(Power_dict['1'])}")
  print(f"Average power usage for ps2 with RL scheduler - Cooperative: {sum(Power_dict['2'])/len(Power_dict['2'])}")
  print(f"Average power usage for ps3 with RL scheduler - Cooperative: {sum(Power_dict['3'])/len(Power_dict['3'])}")
  print(f"Average power usage for ps4 with RL scheduler - Cooperative: {sum(Power_dict['4'])/len(Power_dict['4'])}")
  print(f"Average power usage for ps5 with RL scheduler - Cooperative: {sum(Power_dict['5'])/len(Power_dict['5'])}")
  # print(f"Average power usage for ps6 with RL scheduler - Cooperative: {sum(Power_dict['6'])/Max_Steps[6]}")


  # print(Max_Steps2)
  # print(f"Average power usage for ps1 with RL scheduler - Cooperative: {sum(Power_dict2['1'])/len(Power_dict2['1'])}")
  # print(f"Average power usage for ps2 with RL scheduler - Cooperative: {sum(Power_dict2['2'])/len(Power_dict2['2'])}")
  # print(f"Average power usage for ps3 with RL scheduler - Cooperative: {sum(Power_dict2['3'])/len(Power_dict2['3'])}")
  # print(f"Average power usage for ps4 with RL scheduler - Cooperative: {sum(Power_dict2['4'])/len(Power_dict2['4'])}")
  # print(f"Average power usage for ps5 with RL scheduler - Cooperative: {sum(Power_dict2['5'])/len(Power_dict2['5'])}")
  # print(f"Average power usage for ps6 with RL scheduler - Cooperative: {sum(Power_dict2['6'])/len(Power_dict2['6'])}")

  print(f"Average power usage for ps1 without RL scheduler: {sum(Power_dictRandom['1'])/Max_StepsRandom[1]}")
  print(f"Average power usage for ps2 without RL scheduler: {sum(Power_dictRandom['2'])/Max_StepsRandom[2]}")
  print(f"Average power usage for ps3 without RL scheduler: {sum(Power_dictRandom['3'])/Max_StepsRandom[3]}")
  print(f"Average power usage for ps4 without RL scheduler: {sum(Power_dictRandom['4'])/Max_StepsRandom[4]}")
  
  plt.figure(1)
  plt.title("AoI change with cooperative RL algorithm heuristic reward, Nₖ = 50")
  plt.plot(AoI_dict["1"] , label = "ps1",  linestyle = "dashed")
  plt.plot(AoI_dict["2"] , label = "ps2",  linestyle = "dashed")
  plt.plot(AoI_dict["3"] , label = "ps3",  linestyle = "dashed")
  plt.plot(AoI_dict["4"] , label = "ps4",  linestyle = "dashed")
  plt.plot(AoI_dict["5"] , label = "ps5",  linestyle = "dashed")
  # plt.plot(AoI_dict["6"] , label = "ps5",  linestyle = "dashed")

  plt.axhline(y = 100, color = "orange", linestyle = 'solid', label = "deadline", )
  plt.legend(loc = "best")
 
 
  # plt.figure(2)
  # plt.title("AoI change with cooperative RL algorithm simple reward, Nₖ = 50")
  # plt.plot(AoI_dict2["1"] , label = "ps1",  linestyle = "dashed")
  # plt.plot(AoI_dict2["2"] , label = "ps2",  linestyle = "dashed")
  # plt.plot(AoI_dict2["3"] , label = "ps3",  linestyle = "dashed")
  # plt.plot(AoI_dict2["4"] , label = "ps4",  linestyle = "dashed")
  # plt.plot(AoI_dict2["5"] , label = "ps5",  linestyle = "dashed")
  # plt.plot(AoI_dict2["6"] , label = "ps6",  linestyle = "dashed")

  # plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "deadline", )
  # plt.legend(loc = "best")
 
  plt.figure(3)
  plt.title("AoI change with random selection, Nₖ = 50")
  plt.plot(AoI_dictRandom["1"] , label = "ps1",  linestyle = "dashed")
  plt.plot(AoI_dictRandom["2"] , label = "ps2",  linestyle = "dashed")
  plt.plot(AoI_dictRandom["3"] , label = "ps3", linestyle = "dashed" )
  plt.plot(AoI_dictRandom["4"] , label = "ps4",  linestyle = "dashed")
  plt.plot(AoI_dictRandom["5"] , label = "ps5",  linestyle = "dashed")

  plt.axhline(y = 100, color = "orange", linestyle = 'solid', label = "deadline", )
  plt.legend(loc = "best")

  plt.figure(4)
  plt.title("Power consumption with cooperative RL algorithm heuristic reward, Nₖ = 50")
  plt.plot(Power_dict["1"] , label = "ps1",  linestyle = "dashed")
  plt.plot(Power_dict["2"] , label = "ps2",  linestyle = "dashed")
  plt.plot(Power_dict["3"] , label = "ps3",  linestyle = "dashed")
  plt.plot(Power_dict["4"] , label = "ps4",  linestyle = "dashed")
  plt.plot(Power_dict["5"] , label = "ps5",  linestyle = "dashed")
  plt.legend(loc = "best")

  plt.figure(5)
  plt.title("Power consumption with cooperative RL algorithm simple reward, Nₖ = 50")
  plt.plot(Power_dict2["1"] , label = "ps1",  linestyle = "dashed")
  plt.plot(Power_dict2["2"] , label = "ps2",  linestyle = "dashed")
  plt.plot(Power_dict2["3"] , label = "ps3",  linestyle = "dashed")
  plt.plot(Power_dict2["4"] , label = "ps4",  linestyle = "dashed")
  plt.plot(Power_dict2["5"] , label = "ps5",  linestyle = "dashed")
  plt.legend(loc = "best")

  # plt.figure(3)
  # AOI1_Nk = [np.mean(AoI_dict["1"]), np.mean(AoI_dict2["1"]), np.mean(AoI_dict3["1"]),  np.mean(AoI_dict4["1"])]
  # AOI2_Nk = [np.mean(AoI_dict["2"]), np.mean(AoI_dict2["2"]), np.mean(AoI_dict3["2"]),  np.mean(AoI_dict4["2"])]
  # AOI3_Nk = [np.mean(AoI_dict["3"]), np.mean(AoI_dict2["3"]), np.mean(AoI_dict3["3"]),  np.mean(AoI_dict4["3"])]
  # AOI4_Nk = [np.mean(AoI_dict["4"]), np.mean(AoI_dict2["4"]), np.mean(AoI_dict3["4"]),  np.mean(AoI_dict4["4"])]
  # AOI5_Nk = [np.mean(AoI_dict["5"]), np.mean(AoI_dict2["5"]), np.mean(AoI_dict3["5"]),  np.mean(AoI_dict4["5"])]
  # NK = [3,8,15,25]
  # plt.title(f"Average AoI for physical systems with various number of channel states")
  # plt.plot(NK, AOI1_Nk, label = "ps1", linestyle = "solid", marker = "*")
  # plt.plot(NK, AOI2_Nk, label = "ps2", linestyle = "solid", marker = "*")
  # plt.plot(NK, AOI3_Nk, label = "ps3", linestyle = "solid", marker = "*")
  # plt.plot(NK, AOI4_Nk, label = "ps4", linestyle = "solid", marker = "*")
  # plt.plot(NK, AOI5_Nk, label = "ps5", linestyle = "solid", marker = "*")
  # plt.xlabel("Nₖ")
  # plt.ylabel("Average AoI")
  # plt.legend(loc = "best")

  plt.show()

else:
  break_point = 0
  Results = dict()  
  Results2 = dict()
  Results3 = dict()
  TimeSteps = {"Time":0}
  for ps in range(1,NumberOfPS+1):
    Results[ps] = []
    Results2[ps] = []
    Results3[ps] = []
  if not os.path.exists("./results"):
      os.makedirs("./results")
  if save_models and not os.path.exists("./pytorch_models"):
      os.makedirs("./pytorch_models")

  for ps in range(1,NumberOfPS+1):
    terminal_rewards[ps] = 0
    done[ps] = False
    terminal[ps] = False
    episode_num[ps] = 0
    episode_reward[ps] = 0
    max_episode_reward[ps] = -1000000000000
    min_episode_reward[ps] = +1000000000000
    episode_timesteps[ps] = 0
  timesteps_since_eval = 0
  signal = False
  fig, ax = plt.subplots()
  x_data = []
  y_data = []
  line, = ax.plot(x_data, y_data, c='blue', label='Dynamic Data')
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.set_title('Dynamic Plot')
  ax.legend()
  def update(val):
    x_data.append(total_timesteps)
    y_data.append(val)
    # line.set_data(x_data, y_data) 
    # ax.relim()
    # ax.autoscale_view()
    # plt.pause(0.1)
  while total_timesteps < max_timesteps :
 #   if total_timesteps >= 10000:
#      break
    TimeSteps["Time"] = total_timesteps
    Json1 = json.dumps(TimeSteps)
    f = open("./results/Time.json","w")
    f.write(Json1)
    f.close()
    print(timesteps_since_eval)
    if timesteps_since_eval >= eval_freq:
          timesteps_since_eval %= eval_freq
          res, signal, score , Avg_AoI , Avg_power = evaluate_policy(Env= Env, policy= policy)
          # update(res[1])
          if signal == True:
            break_point += 1
          if break_point == 10:
            break
          Evaluations.append(score)
          for ps in range(1,NumberOfPS+1):
            Results[ps].append(res[ps])
          Json = json.dumps(Results)
          f = open("./results/Results.json","w")
          f.write(Json)
          f.close()
          for ps in range(1,NumberOfPS+1):
            Results2[ps].append(Avg_AoI[ps])
          Json2 = json.dumps(Results2)
          f = open("./results/Results2.json","w")
          f.write(Json2)
          f.close()
          for ps in range(1,NumberOfPS+1):
            Results3[ps].append(Avg_power[ps])
          Json3 = json.dumps(Results3)
          f = open("./results/Results3.json","w")
          f.write(Json3)
          f.close()
    for ps in range(1,NumberOfPS+1):
      if total_timesteps%1 == 0 and total_timesteps > 200:
        central_critic = policy[ps].train(central_critic, 1 , policy, replay_buffer, ps, batch_size, discount, tau, noise_clip, policy_freq)
        policy[ps].save(f"{ps}th PS", "./pytorch_models")
      if episode_timesteps[ps] >= 200:
        if total_timesteps != 0:
          print(f"PS: {ps}, Total Timesteps: {total_timesteps}, Episode Num: {episode_num[ps]}, Reward: {episode_reward[ps]}")
          # print(central_critic.Critic.state_dict())
          states[ps] = Env.reset(ps= ps)
          done[ps] = False
          if episode_reward[ps] > max_episode_reward[ps]:
              max_episode_reward[ps] = episode_reward[ps]
          elif episode_reward[ps] < min_episode_reward[ps]:
              min_episode_reward[ps] = episode_reward[ps]
          episode_reward[ps] = 0
          episode_timesteps[ps] = 0
          episode_num[ps] += 1

      if total_timesteps < start_timesteps:
         actions[ps] = np.random.uniform(0,policy[ps].max_action,1)
#        print(actions[ps])
      else:
#         if np.random.rand() < eps:
#            actions[ps] = np.random.uniform(0,policy[ps].max_action,1)
#         else:
         actions[ps] = policy[ps].select_action(states[ps])
         actions[ps] = abs(actions[ps])
         actions[ps] = (actions[ps] + np.random.normal(0,expl_noise,1)).clip(0, policy[ps].max_action) 
        
#actions[ps] = abs(add_gaussian_noise(actions[ps]))
#             actions[ps] = abs(actions[ps])
    next_states, rewards, done, terminal = Env.step(actions, total_timesteps)  
#    print(rewards)
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
    }
    for ps in range(1,NumberOfPS+1):
      episode_reward[ps] += rewards[ps]
#      shaped_reward = 1 + ((episode_reward[ps] - max_episode_reward[ps])/(max_episode_reward[ps] - min_episode_reward[ps]))
      shaped_reward = 0 
      if done[ps] == 1:
        if terminal[ps] == 1 or episode_timesteps[ps] == 200:
          replay_buffer[ps].add((states[ps], actions[ps], next_states[ps], rewards[ps], terminal[ps], shaped_reward))
      else:
        replay_buffer[ps].add((states[ps], actions[ps],next_states[ps], rewards[ps], terminal[ps], shaped_reward))
      episode_timesteps[ps] += 1
    

    timesteps_since_eval += 1

    states = next_states
    total_timesteps += 1
    epsilon = 5e-5
    if total_timesteps > start_timesteps:
       eps = eps - delta_eps if  eps > eps_min else eps_min
       expl_noise = expl_noise - epsilon if expl_noise > expl_noise_min else expl_noise_min
