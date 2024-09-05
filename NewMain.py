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



StepSize = SimulationParams.TimestepSize
NumberOfPS = SimulationParams.NumberOfPS
NumOfGainStates = SimulationParams.NumOfGainStates
def add_gaussian_noise(action, expl_noise =0.002, min_action=0, max_action=1.0):
    noise = np.random.normal(0, max_action * expl_noise, size= 1)
    noisy_action = action + noise
    clipped_action = np.clip(noisy_action, min_action, max_action)
    return clipped_action
AoiViolation = []

def evaluate_policy(Env, policy, eval_episodes = 20):
  signal = False
  scores = dict()
  cumulative = dict()
  avg_reward = dict()
  AoIs = dict()
  powers = dict()
  cr_reward = dict()
  action = dict()
  states = dict()
  AoIErrorDict = dict()
  PowerUsageDict = dict()
  AoI_Violoation_Probability = {}

  for i in range(1,NumberOfPS+1):
     avg_reward[i] = 0 
     AoIErrorDict[i]  = 0 
     PowerUsageDict[i] = 0
     AoIs[i] = 0
     powers[i] = 0
     cr_reward[i] = 0
     scores[i] = 0
  for j in range(eval_episodes):

    time = 0
    timesteps = 0
    for i in range(1,NumberOfPS+1):

      cr_reward[i] = 0
      states[i]= Env.reset(ps = i)
      cumulative[i] = 0 

    while time < 200:
      timesteps += StepSize
      for i in range(1,NumberOfPS+1):
        if timesteps > 200: 
          timesteps = 0 
          states[ps] = Env.reset(ps= i)
          done[ps] = False
        action[i] = policy[i].select_action(states[i])
        action[i] = abs(action[i])
         
      # states, rewards, dones, terminal = Env.step(action = action, time = time)
      states, rewards, dones, terminal = Env.stepWithStepSize(action= action, time= time, stepSize = StepSize)

      states = {
          i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
      }
      time += StepSize
      for i in range(1,NumberOfPS+1):
        avg_reward[i] += rewards[i]
#        AoIs[i] += states[i][1] * SimulationParams.deadlines[0]
        AoIs[i] += states[i][1]     
        powers[i] += action[i][0]
        if states[i][1] > 80: 
          AoIErrorDict[i] += 1
        PowerUsageDict[i] += action[i][0]
        cr_reward[i] += rewards[i]
    for i in range(1,NumberOfPS+1):
      scores[i] = scores[i] + 1 if cr_reward[i] > -100 else scores[i]
  for i in range(1,NumberOfPS+1):
      AoIErrorDict[i] = int((AoIErrorDict[i]/((200*eval_episodes)/StepSize))*(100))
      PowerUsageDict[i] = ((PowerUsageDict[i]/((200*eval_episodes)/StepSize)))

      avg_reward[i] = avg_reward[i]*StepSize/eval_episodes

      AoIs[i] = AoIs[i]*StepSize/eval_episodes 
      powers[i] = powers[i]*StepSize/eval_episodes
      scores[i] = scores[i]*StepSize/eval_episodes
  signal = False
 
  averageAoIViolation =  (np.mean(list((AoIErrorDict.values()))))
  AoiViolation.append(averageAoIViolation)
  averagePowerUsage =  (np.mean(list((PowerUsageDict.values()))))
  if averageAoIViolation < 5 and averagePowerUsage < 0.2:
    signal = True
  return avg_reward, signal , scores, AoIs , powers, AoiViolation
    
# def evaluate_policy(Env, policy, eval_episodes = 20):
#   signal = False
#   scores = dict()
#   cumulative = dict()
#   avg_reward = dict()
#   AoIs = dict()
#   powers = dict()
#   cr_reward = dict()
#   action = dict()
#   states = dict()
#   for i in range(1,NumberOfPS+1):
#      avg_reward[i] = 0 
#      AoIs[i] = 0
#      powers[i] = 0
#      cr_reward[i] = 0
#      scores[i] = 0
#   for j in range(eval_episodes):
#     time = 0
#     for i in range(1,NumberOfPS+1):
#       cr_reward[i] = 0
#       states[i]= Env.reset(ps = i)
#       cumulative[i] = 0 

#     while time < 200:
#       for i in range(1,NumberOfPS+1):

#         action[i] = policy[i].select_action(states[i])
#         action[i] = abs(action[i])
#       states, rewards, dones, terminal = Env.stepWithStepSize(action= action, time= time, stepSize = StepSize)
#       states = {
#           i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
#           for i in range(1, NumberOfPS + 1)
#       }
#       time += StepSize
#       for i in range(1,NumberOfPS+1):
#         avg_reward[i] += rewards[i]
# #        AoIs[i] += states[i][1] * SimulationParams.deadlines[0]
#         AoIs[i] += states[i][1]     
#         powers[i] += action[i][0]
#         cr_reward[i] += rewards[i]
#     for i in range(1,NumberOfPS+1):
#       scores[i] = scores[i] + 1 if cr_reward[i] > -100 else scores[i]
#   for i in range(1,NumberOfPS+1):
#       avg_reward[i] = (avg_reward[i]*StepSize)/eval_episodes
#       AoIs[i] = (AoIs[i]*StepSize)/eval_episodes 
#       powers[i] = (powers[i]*StepSize)/eval_episodes
#       scores[i] = (scores[i]*StepSize)/eval_episodes
#   signal = True
#   for i in range(1,NumberOfPS+1):
#     if avg_reward[i] < -300:
#         signal = False
#         break
#   return avg_reward, signal , scores, AoIs , powers
save_models = True
expl_noise_min = 0.01
epsilon = 5e-4
eval_freq = 500
batch_size = 256
tau = 0.005
discount = 0.99
expl_noise = 0.1
eps = 0.5
delta_eps = 2e-5
eps_min = 0.02
noise_clip =  0.5
policy_freq = 2
max_timesteps = 20e5
start_timesteps = 20e3
total_timesteps = 0
episode_num = dict()
episode_reward = dict()
min_episode_reward = dict()
max_episode_reward = dict()
episode_timesteps = dict()
timesteps_since_eval = dict()
done = dict()
is_completely_done = dict()
rewards = dict()
terminal_rewards = dict()
terminal = dict()
states = dict()
actions = dict()
next_states = dict()
Env = CustumEnv(NumberOfPS, NumOfGainStates)
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
  policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  replay_buffer[i] = Replay_Buffer()

central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, action_dim=  SimulationParams.NumberOfTCh)


if Test == True:

  policy[1].load(f"{1}th PS", f"./pytorch_models")
  policy[2].load(f"{2}th PS", f"./pytorch_models")
  policy[3].load(f"{3}th PS", f"./pytorch_models")
  policy[4].load(f"{4}th PS", f"./pytorch_models")
  policy[5].load(f"{5}th PS", f"./pytorch_models")
  # policy[6].load(f"{6}th PS", f"./pytorch_models/5ps_100ch_centralcritic_100bits_40w_80d_power_reward_powermonitor_paper")


  Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0} 
  Max_Steps2 = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0} 
  Max_Steps3 = {1:0, 2:0, 3:0 , 4:0 ,5:0, 6:0} 
  Max_Steps4 = {1:0, 2:0, 3:0 , 4:0 ,5:0, 6:0} 
  Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0} 


  max_Steps = 0
  AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[]}
  Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[]}



  while total_timesteps < max_timesteps:
    for ps in range(1,NumberOfPS+1):
      actions[ps]  = policy[ps].select_action(states[ps])
      actions[ps] = abs(actions[ps])
    next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)          
    }
    for ps in range(1, NumberOfPS + 1):
      if next_states[ps][2] != 0 :
        Max_Steps[ps] += 1  
    

    for ps in range(1,NumberOfPS+1):
      Power_dict[f"{ps}"].append(actions[ps])
      AoI_dict[f"{ps}"].append(states[ps][1])
      Bits_Dict[f"{ps}"].append(states[ps][2])
    states = next_states
    total_timesteps += 1
  print(f"Average power usage for ps1 with RL scheduler - Cooperative: {sum(Power_dict['1'])/len(Power_dict['1'])}")
  print(f"Average power usage for ps2 with RL scheduler - Cooperative: {sum(Power_dict['2'])/len(Power_dict['2'])}")
  print(f"Average power usage for ps3 with RL scheduler - Cooperative: {sum(Power_dict['3'])/len(Power_dict['3'])}")
  print(f"Average power usage for ps4 with RL scheduler - Cooperative: {sum(Power_dict['4'])/len(Power_dict['4'])}")
  print(f"Average power usage for ps5 with RL scheduler - Cooperative: {sum(Power_dict['5'])/len(Power_dict['5'])}")
  


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
  plt.figure(4)
  plt.title("Power consumption with cooperative RL algorithm heuristic reward, Nₖ = 50")
  plt.plot(Power_dict["1"] , label = "ps1",  linestyle = "dashed")
  plt.plot(Power_dict["2"] , label = "ps2",  linestyle = "dashed")
  plt.plot(Power_dict["3"] , label = "ps3",  linestyle = "dashed")
  plt.plot(Power_dict["4"] , label = "ps4",  linestyle = "dashed")
  plt.plot(Power_dict["5"] , label = "ps5",  linestyle = "dashed")
  plt.legend(loc = "best")
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
  while total_timesteps < max_timesteps :
    if timesteps_since_eval >= eval_freq:
          TimeSteps["Time"] = total_timesteps
          Json1 = json.dumps(TimeSteps)
          f = open("./results/Time.json","w")
          f.write(Json1)
          f.close()
          timesteps_since_eval %= eval_freq
          res, signal, score , Avg_AoI , Avg_power, Avg_AoIViolation = evaluate_policy(Env= Env, policy= policy)
          if signal == True:
            break_point += 1
          if break_point == 5:
            break
          Jsonviolation = json.dumps(Avg_AoIViolation)
          f = open("./results/Results4.json","w")
          f.write(Jsonviolation)
          f.close()
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
      if total_timesteps%200 == 0: 
        print(actions)
      if total_timesteps%50 == 0 and total_timesteps > 200:
        policy[ps].save(f"{ps}th PS", "./pytorch_models")
      if episode_timesteps[ps] >= 200:
        if total_timesteps != 0:
          states[ps] = Env.reset(ps= ps)
          done[ps] = False
          episode_reward[ps] = 0
          episode_timesteps[ps] = 0
          episode_num[ps] += 1

      if total_timesteps < start_timesteps:
         actions[ps] = np.random.uniform(0,policy[ps].max_action,1)
      # print(actions[ps])
      # if ps in  [2,1,3,4]:
      #   actions[ps] = np.array([0])
      # actions[5] = np.array([1])
      else:
         actions[ps] = policy[ps].select_action(states[ps])
         actions[ps] = abs(actions[ps])
         actions[ps] = (actions[ps] + np.random.normal(0,expl_noise,1)).clip(0, policy[ps].max_action) 
        
    next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, StepSize)  
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
    }
    # print(next_states)
    # print(terminal)
    # print(total_timesteps)
    # T.sleep(2)
    for ps in range(1,NumberOfPS+1):
      episode_reward[ps] += rewards[ps]
      shaped_reward = 0 
      replay_buffer[ps].add((states[ps], actions[ps],next_states[ps], rewards[ps], terminal[ps], shaped_reward))
      episode_timesteps[ps] += 1
    timesteps_since_eval += StepSize
    states = next_states
    total_timesteps += StepSize
    epsilon = 5e-5
    if total_timesteps > start_timesteps:
       eps = eps - delta_eps if  eps > eps_min else eps_min
       expl_noise = expl_noise - epsilon if expl_noise > expl_noise_min else expl_noise_min
