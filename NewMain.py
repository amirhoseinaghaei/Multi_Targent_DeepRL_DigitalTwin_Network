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

def evaluate_policy(Env, policy, eval_episodes = 10):
  signal = False
  scores = dict()
  cumulative = dict()
  avg_reward = dict()
  cr_reward = dict()
  action = dict()
  states = dict()
  for i in range(1,NumberOfPS+1):
     avg_reward[i] = 0 
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
        cr_reward[i] += rewards[i]
    for i in range(1,NumberOfPS+1):
      scores[i] = scores[i] + 1 if cr_reward[i] > -100 else scores[i]
  for i in range(1,NumberOfPS+1):
      avg_reward[i] = avg_reward[i]/eval_episodes
      scores[i] = scores[i]/10
  signal = True
  for i in range(1,NumberOfPS+1):
    if avg_reward[i] <= -50:

        signal = False
        break
  return avg_reward, signal , scores
save_models = True
expl_noise_min = 0.001
epsilon = 5e-4
expl_noise = 0.002
eval_freq = 200
batch_size = 100
tau = 0.005
discount = 0.99
noise_clip =  0.005
policy_noise = 0.005
policy_freq = 2
max_timesteps = 4e4
start_timesteps = 3e3
total_timesteps = 0
episode_num = dict()
episode_reward = dict()
episode_timesteps = dict()
timesteps_since_eval = dict()
done = dict()
done2 = dict()
done3 = dict()
doneRandom = dict()
is_completely_done = dict()
rewards = dict()
rewards2 = dict()
rewards3 = dict()
rewardsRandom = dict()
terminal_rewards = dict()
terminal = dict()
terminal_rewards2 = dict()
terminal_rewards3 = dict()
terminal2 = dict()
terminal3 = dict()
terminalRandom = dict()
states = dict()
actions = dict()
next_states = dict()
states2 = dict()
actions2 = dict()
states3 = dict()
actions3 = dict()
actionsRandom = dict()
statesRandom = dict()
next_states2 = dict()
next_states3 = dict()
next_statesRandom = dict()
Env = CustumEnv(NumberOfPS)
Env2 = CustumEnv(NumberOfPS)
Env3 = CustumEnv(NumberOfPS)
EnvRandom = CustumEnv(NumberOfPS)
Test = False
model = 9
policy = dict()
policy2 = dict()
policy3 = dict()
new_policy = dict()
policyRandom = dict()
Evaluations = []
Reward_Evaluations = []
replay_buffer = dict()

for i in range(1,NumberOfPS+1):
  states[i] = Env.reset(i)
  states2[i] = Env2.reset(i)
  states3[i] = Env3.reset(i)
  statesRandom[i] = EnvRandom.reset(i)

  policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= 1, max_action= 1)
  policy2[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= 1, max_action= 1)
  policy3[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= 1, max_action= 1)
  replay_buffer[i] = Replay_Buffer()

central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, action_dim= 1)

if Test == True:

  policy[1].load(f"{1}th PS", f"./pytorch_models")
  policy[2].load(f"{2}th PS", f"./pytorch_models")
  policy[3].load(f"{3}th PS", f"./pytorch_models")
  policy[4].load(f"{4}th PS", f"./pytorch_models")
  # policy[5].load(f"{5}th PS", f"./pytorch_models")
  # policy[6].load(f"{6}th PS", f"./pytorch_models")

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
  Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0} 


  max_Steps = 0
  AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dict2 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict2 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dict3 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict3 = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  AoI_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_Dict2 = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_Dict3 = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}
  Bits_DictRandom = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}

  while total_timesteps < max_timesteps:
    for ps in range(1,NumberOfPS+1):
      actions[ps]  = policy[ps].select_action(states[ps])
      actions[ps] = abs(actions[ps])
      # actions2[ps]  = policy2[ps].select_action(states2[ps])
      # actions2[ps] = abs(actions2[ps])
      # actions3[ps]  = policy3[ps].select_action(states3[ps])
      # actions3[ps] = abs(actions3[ps])
      if statesRandom[ps][2] <= 100:
        actionsRandom[ps] = np.array([0])
      else:
          actionsRandom[ps] = np.array([np.random.uniform(0,1)])
    next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
    # next_states2, rewards2, done2, terminal2 = Env2.step(actions2, total_timesteps)    
    # next_states3, rewards3, done3, terminal3 = Env3.step(actions3, total_timesteps)    
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
      Power_dictRandom[f"{ps}"].append(actionsRandom[ps])
      AoI_dictRandom[f"{ps}"].append(statesRandom[ps][1])
      Bits_DictRandom[f"{ps}"].append(statesRandom[ps][2])

  
    states = next_states
    # states2 = next_states2
    # states3 = next_states3
    statesRandom = next_statesRandom

    total_timesteps += 1
  print(Max_Steps)
  print(f"Average power usage for ps1 with RL scheduler - Cooperative: {sum(Power_dict['1'])/Max_Steps[1]}")
  print(f"Average power usage for ps2 with RL scheduler - Cooperative: {sum(Power_dict['2'])/Max_Steps[2]}")
  print(f"Average power usage for ps3 with RL scheduler - Cooperative: {sum(Power_dict['3'])/Max_Steps[3]}")
  print(f"Average power usage for ps4 with RL scheduler - Cooperative: {sum(Power_dict['4'])/Max_Steps[4]}")
  # print(f"Average power usage for ps5 with RL scheduler - Cooperative: {sum(Power_dict['5'])/Max_Steps[5]}")
  # # print(f"Average power usage for ps6 with RL scheduler - Cooperative: {sum(Power_dict['6'])/Max_Steps[6]}")

  # print(f"Average power usage for ps1 with RL scheduler - Cooperative with power optimization: {sum(Power_dict3['1'])/Max_Steps3[1]}")
  # print(f"Average power usage for ps2 with RL scheduler - Cooperative with power optimization: {sum(Power_dict3['2'])/Max_Steps3[2]}")
  # print(f"Average power usage for ps3 with RL scheduler - Cooperative with power optimization: {sum(Power_dict3['3'])/Max_Steps3[3]}")
  # print(f"Average power usage for ps4 with RL scheduler - Cooperative with power optimization: {sum(Power_dict3['4'])/Max_Steps3[4]}")
  # # print(f"Average power usage for ps5 with RL scheduler - Cooperative with power optimization: {sum(Power_dict3['5'])/Max_Steps3[5]}")

  # print(f"Average power usage for ps1 with RL scheduler -  Non-Cooperative: {sum(Power_dict2['1'])/Max_Steps2[1]}")
  # print(f"Average power usage for ps2 with RL scheduler -  Non-Cooperative: {sum(Power_dict2['2'])/Max_Steps2[2]}")
  # print(f"Average power usage for ps3 with RL scheduler -  Non-Cooperative: {sum(Power_dict2['3'])/Max_Steps2[3]}")
  # print(f"Average power usage for ps4 with RL scheduler -  Non-Cooperative: {sum(Power_dict2['4'])/Max_Steps2[4]}")
  # # print(f"Average power usage for ps5 with RL scheduler -  Non-Cooperative: {sum(Power_dict2['5'])/Max_Steps2[5]}")

  print(f"Average power usage for ps1 without RL scheduler: {sum(Power_dictRandom['1'])/Max_StepsRandom[1]}")
  print(f"Average power usage for ps2 without RL scheduler: {sum(Power_dictRandom['2'])/Max_StepsRandom[2]}")
  print(f"Average power usage for ps3 without RL scheduler: {sum(Power_dictRandom['3'])/Max_StepsRandom[3]}")
  print(f"Average power usage for ps4 without RL scheduler: {sum(Power_dictRandom['4'])/Max_StepsRandom[4]}")
  # print(f"Average power usage for ps5 without RL scheduler: {sum(Power_dictRandom['5'])/Max_StepsRandom[5]}")
  # # print(f"Average power usage for ps6 without RL scheduler: {sum(Power_dictRandom['6'])/Max_StepsRandom[6]}")

  # print(f"Average AoI for ps1 with RL scheduler - Cooperative: {sum(AoI_dict['1'])/Max_Steps[1]}")
  # print(f"Average AoI for ps2 with RL scheduler - Cooperative: {sum(AoI_dict['2'])/Max_Steps[2]}")
  # print(f"Average AoI for ps3 with RL scheduler - Cooperative: {sum(AoI_dict['3'])/Max_Steps[3]}")
  # print(f"Average AoI for ps4 with RL scheduler - Cooperative: {sum(AoI_dict['4'])/Max_Steps[4]}")

  # print(f"Average AoI for ps1 with RL scheduler - Non-Cooperative: {sum(AoI_dict2['1'])/Max_Steps2[1]}")
  # print(f"Average AoI for ps2 with RL scheduler - Non-Cooperative: {sum(AoI_dict2['2'])/Max_Steps2[2]}")
  # print(f"Average AoI for ps3 with RL scheduler - Non-Cooperative: {sum(AoI_dict2['3'])/Max_Steps2[3]}")
  # print(f"Average AoI for ps4 with RL scheduler - Non-Cooperative: {sum(AoI_dict2['4'])/Max_Steps2[4]}")

  # print(f"Average AoI for ps1 without RL scheduler: {sum(AoI_dictRandom['1'])/Max_StepsRandom[1]}")
  # print(f"Average AoI for ps2 without RL scheduler: {sum(AoI_dictRandom['2'])/Max_StepsRandom[2]}")
  # print(f"Average AoI for ps3 without RL scheduler: {sum(AoI_dictRandom['3'])/Max_StepsRandom[3]}")
  # print(f"Average AoI for ps4 without RL scheduler: {sum(AoI_dictRandom['4'])/Max_StepsRandom[4]}")

  plt.figure(1)
  plt.title("AoI change ps1")
  plt.plot(AoI_dict["1"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["1"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["1"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["1"] , color = "red", label = "Random selection")
  plt.axhline(y = 40, color = "orange", linestyle = 'solid', label = "deadline ps1", )
  plt.legend(loc = "best")

  plt.figure(2)
  plt.title("AoI change ps2")
  plt.plot(AoI_dict["2"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["2"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["2"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["2"] , color = "red", label = "Random selection")
  plt.axhline(y = 40, color = "orange", linestyle = 'solid', label = "deadline ps2", )
  plt.legend(loc = "best")

  plt.figure(3)
  plt.title("AoI change ps3")
  plt.plot(AoI_dict["3"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["3"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["3"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["3"] , color = "red", label = "Random selection")
  plt.axhline(y = 40, color = "orange", linestyle = 'solid', label = "deadline ps3", )
  plt.legend(loc = "best")

  plt.figure(4)
  plt.title("AoI change ps4")
  plt.plot(AoI_dict["4"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["4"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["4"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["4"] , color = "red", label = "Random selection")
  plt.axhline(y = 40, color = "orange", linestyle = 'solid', label = "deadline ps4", )
  plt.legend(loc = "best")


  plt.figure(5)
  plt.title("AoI change ps5")
  plt.plot(AoI_dict["5"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["4"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["5"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["5"] , color = "red", label = "Random selection")
  plt.axhline(y = 50, color = "orange", linestyle = 'solid', label = "deadline ps5", )
  plt.legend(loc = "best")
  
  plt.figure(6)
  plt.title("AoI change ps6")
  plt.plot(AoI_dict["6"] , color = "green", label = "With RL Scheduler - cooperative")
  # plt.plot(AoI_dict3["4"] , color = "black", label = "With RL Scheduler - cooperative - with power")
  # plt.plot(AoI_dict2["5"] , color = "blue", label = "With RL Scheduler - Non-cooperative")
  plt.plot(AoI_dictRandom["6"] , color = "red", label = "Random selection")
  plt.axhline(y = 50, color = "orange", linestyle = 'solid', label = "deadline ps6", )
  plt.legend(loc = "best")
  # plt.figure(5)
  # plt.title("Power usage of PS1")
  # plt.plot(Power_dict["1"] , color = "green", label = "Cooperative")
  # plt.plot(Power_dict3["1"] , color = "black", label = "Cooperative - power")
  # plt.plot(Power_dict2["1"] , color = "blue", label = "Non-Cooperative")
  # plt.plot(Power_dictRandom["1"] , color = "red", label = "Random")

  # plt.figure(6)
  # plt.title("Power usage of PS2")
  # plt.plot(Power_dict["2"] , color = "green", label = "Cooperative")
  # plt.plot(Power_dict3["2"] , color = "black", label = "Cooperative - power")
  # plt.plot(Power_dict2["2"] , color = "blue", label = "Non-Cooperative")
  # plt.plot(Power_dictRandom["2"] , color = "red", label = "Random")
  # plt.figure(7)
  # plt.title("Power usage of PS3")
  # plt.plot(Power_dict["3"] , color = "green", label = "Cooperative")
  # plt.plot(Power_dict3["3"] , color = "black", label = "Cooperative - power")
  # plt.plot(Power_dict2["3"] , color = "blue", label = "Non-Cooperative")
  # plt.plot(Power_dictRandom["3"] , color = "red", label = "Random")

  # plt.figure(8)
  # plt.title("Power usage of PS4")
  # plt.plot(Power_dict["4"] , color = "green", label = "Cooperative")
  # plt.plot(Power_dict3["4"] , color = "black", label = "Cooperative - power")
  # plt.plot(Power_dict2["4"] , color = "blue", label = "Non-Cooperative")
  # plt.plot(Power_dictRandom["4"] , color = "red", label = "Random")



  plt.figure(9)
  plt.title("Bits with RL power scheduler")
  plt.plot(Bits_Dict["1"],  color = "green", label = "ps1")
  plt.plot(Bits_Dict["2"],  color = "blue", label = "ps2")
  plt.plot(Bits_Dict["3"],  color = "black", label = "ps3")
  plt.plot(Bits_Dict["4"],  color = "red", label = "ps4")
  plt.plot(Bits_Dict["5"],  color = "aqua", label = "ps5")
  plt.legend(loc = "best")

  # plt.figure(10)
  # plt.title("Bits with RL power scheduler - Cooperative - with power")
  # plt.plot(Bits_Dict3["1"],  color = "green", label = "ps1")
  # plt.plot(Bits_Dict3["2"],  color = "blue", label = "ps2")
  # plt.plot(Bits_Dict3["3"],  color = "black", label = "ps3")
  # plt.plot(Bits_Dict3["4"],  color = "red", label = "ps4")
  # plt.legend(loc = "best")

  # plt.figure(11)
  # plt.title("Bits with RL power scheduler -  Non-Cooperative")
  # plt.plot(Bits_Dict2["1"],  color = "green", label = "ps1")
  # plt.plot(Bits_Dict2["2"],  color = "blue", label = "ps2")
  # plt.plot(Bits_Dict2["3"],  color = "black", label = "ps3")
  # plt.plot(Bits_Dict2["4"],  color = "red", label = "ps4")
  # plt.legend(loc = "best")
  # plt.figure(12)
  # plt.title("Bits without RL power scheduler")
  # plt.plot(Bits_DictRandom["1"],  color = "green", label = "ps1")
  # plt.plot(Bits_DictRandom["2"],  color = "blue", label = "ps2")
  # plt.plot(Bits_DictRandom["3"],  color = "black", label = "ps3")
  # plt.plot(Bits_DictRandom["4"],  color = "red", label = "ps4")
  # plt.legend(loc = "best")

  plt.show()

else:
  Results = dict()  
  TimeSteps = {"Time":0}
  for ps in range(1,NumberOfPS+1):
    Results[ps] = []
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
    line.set_data(x_data, y_data) 
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.1)
  while total_timesteps < max_timesteps :
    TimeSteps["Time"] = total_timesteps
    Json1 = json.dumps(TimeSteps)
    f = open("./results/Time.json","w")
    f.write(Json1)
    f.close()
    print(timesteps_since_eval)
    if timesteps_since_eval >= eval_freq:
          timesteps_since_eval %= eval_freq
          res, signal, score = evaluate_policy(Env= Env, policy= policy)
          update(res[1])
          print(signal)
          Evaluations.append(score)
          for ps in range(1,NumberOfPS+1):
            Results[ps].append(res[ps])
          Json = json.dumps(Results)
          f = open("./results/Results.json","w")
          f.write(Json)
          f.close()
    for ps in range(1,NumberOfPS+1):
      if total_timesteps > 400:
        central_critic = policy[ps].train(central_critic, 200, policy, replay_buffer, ps, batch_size, discount, tau, noise_clip, policy_freq)
        policy[ps].save(f"{ps}th PS", "./pytorch_models")
      if episode_timesteps[ps] >= 200:
        if total_timesteps != 0:
          print(f"PS: {ps}, Total Timesteps: {total_timesteps}, Episode Num: {episode_num[ps]}, Reward: {episode_reward[ps]}")
          # print(central_critic.Critic.state_dict())
          states[ps] = Env.reset(ps= ps)
          done[ps] = False
          episode_reward[ps] = 0
          episode_timesteps[ps] = 0
          episode_num[ps] += 1

      if total_timesteps < start_timesteps:
        actions[ps] = np.random.uniform(0,policy[ps].max_action,1)

      else:
          actions[ps] = policy[ps].select_action(states[ps])
          actions[ps] = abs(actions[ps])
          print(f"PS: {ps}, Action: {actions[ps]}")
          actions[ps] = (actions[ps] + np.random.normal(0,expl_noise,1)).clip(0, policy[ps].max_action) 
          # actions[ps] = abs(actions[ps])

    next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
    }
    for ps in range(1,NumberOfPS+1):
      episode_reward[ps] += rewards[ps]
      if done[ps] == 1:
        if terminal[ps] == 1 or episode_timesteps[ps] == 200:
          replay_buffer[ps].add((states[ps], actions[ps], next_states[ps], rewards[ps], done[ps]))
      else:
        replay_buffer[ps].add((states[ps], actions[ps],next_states[ps], rewards[ps], done[ps]))
      episode_timesteps[ps] += 1
    timesteps_since_eval += 1

    states = next_states
    total_timesteps += 1
    expl_noise = expl_noise - epsilon if expl_noise > expl_noise_min else expl_noise_min

  plt.show()
