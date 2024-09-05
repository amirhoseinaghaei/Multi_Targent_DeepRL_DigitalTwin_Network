import copy
import itertools
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
from MDDPG.NMTD3 import TD3 
import time as T
from matplotlib import pyplot as plt
from MDDPG.Replay_Buffer  import Replay_Buffer
from MDDPG.Central_Critic import Central_Critic
import torch
import os
import json




NumberOfPS = SimulationParams.NumberOfPS
NumOfGainStates = SimulationParams.NumOfGainStates
TimestepSize = SimulationParams.TimestepSize






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
      timesteps += TimestepSize
      for i in range(1,NumberOfPS+1):
        if timesteps > 200: 
          timesteps = 0 
          states[ps] = Env.reset(ps= i)
          done[ps] = False
        action[i] = policy[i].select_action(states[i])
        action[i] = abs(action[i])
         
      states, rewards, dones, terminal = Env.stepWithStepSize(action = action, time = time, stepSize = TimestepSize)
      states = {
          i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
          for i in range(1, NumberOfPS + 1)
      }
      time += TimestepSize
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
      AoIErrorDict[i] = int((AoIErrorDict[i]/((200*eval_episodes)/TimestepSize))*(100))
      PowerUsageDict[i] = ((PowerUsageDict[i]/((200*eval_episodes)/TimestepSize)))

      avg_reward[i] = avg_reward[i]*TimestepSize/eval_episodes

      AoIs[i] = AoIs[i]*TimestepSize/eval_episodes 
      powers[i] = powers[i]*TimestepSize/eval_episodes
      scores[i] = scores[i]*TimestepSize/eval_episodes
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
         
#       states, rewards, dones, terminal = Env.stepWithStepSize(action = action, time = time,  stepSize = TimestepSize)
#       states = {
#           i: np.array(list(states[0][f"ps{i}"]) + [states[1][f"ps{i}"]] + [states[2][f"ps{i}"]])
#           for i in range(1, NumberOfPS + 1)
#       }
#       time += TimestepSize
#       for i in range(1,NumberOfPS+1):
#         avg_reward[i] += rewards[i]
# #        AoIs[i] += states[i][1] * SimulationParams.deadlines[0]
#         AoIs[i] += states[i][1]     
#         powers[i] += action[i][0]
#         cr_reward[i] += rewards[i]
#     for i in range(1,NumberOfPS+1):
#       scores[i] = scores[i] + 1 if cr_reward[i] > -100 else scores[i]
#   for i in range(1,NumberOfPS+1):
#       avg_reward[i] = avg_reward[i]*TimestepSize/eval_episodes
#       AoIs[i] = AoIs[i]*TimestepSize/eval_episodes 
#       powers[i] = powers[i]*TimestepSize/eval_episodes
#       scores[i] = scores[i]*TimestepSize/eval_episodes
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
max_timesteps = 10e5
start_timesteps = 100e3
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
actionsRandom = dict()
statesRandom = dict()
rewardsRandom = dict()
doneRandom = dict()
next_statesRandom = dict()
Env = CustumEnv(NumberOfPS, NumOfGainStates)
EnvRandom = CustumEnv(NumberOfPS, NumOfGainStates)
Test = False
policy = dict()
policyRandom = dict()
new_policy = dict()
policyRandom = dict()
Evaluations = []
Reward_Evaluations = []
replay_buffer = dict()
TestAoIDict = dict()
TestTimeDict = dict()
for ps in range(1, NumberOfPS + 1):
    TestAoIDict[ps] = []
    TestTimeDict[ps] = []

for i in range(1,NumberOfPS+1):
  states[i] = Env.reset(i)
  statesRandom[i] = EnvRandom.reset(i)
  policy[i] = TD3(state_dim= SimulationParams.NumberOfTCh + 2, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
  replay_buffer[i] = Replay_Buffer()

# central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, action_dim=  SimulationParams.NumberOfTCh)


if Test == True:

  for i in range(1,SimulationParams.NumberOfPS+1):
    policy[i].load(f"{i}th PS", f"./results/Final/{SimulationParams.NumberOfPS}ps_{SimulationParams.windows[0]}_{SimulationParams.windows[0]*2}_{SimulationParams.NumberOfBits[0]}_{TimestepSize}GR")


  Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0,7:0, 8:0,9:0} 
  Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0,7:0, 8:0,9:0} 


  max_Steps = 0
  AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
  Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
  Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
  AoI_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[],"7":[], "8":[], "9":[]}
  Power_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
  Bits_DictRandom = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}


  while total_timesteps < max_timesteps:
    for ps in range(1,NumberOfPS+1):
      actions[ps]  = policy[ps].select_action(states[ps])
      actions[ps] = abs(actions[ps])
      if statesRandom[ps][2] <= 100:
        actionsRandom[ps] = np.array([0])
      else:
        actionsRandom[ps] = np.array([np.random.uniform(0,1)]) 
    # next_states, rewards, done, terminal = Env.step(actions, total_timesteps)  
    next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, TimestepSize)  

    next_statesRandom, rewardsRandom, doneRandom, terminalRandom = EnvRandom.stepWithStepSize(actionsRandom, total_timesteps, TimestepSize)     
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)          
    }
    next_statesRandom = {
            i: np.array(list(next_statesRandom[0][f"ps{i}"]) + [next_statesRandom[1][f"ps{i}"]] + [next_statesRandom[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
            
    }
    for ps in range(1, NumberOfPS + 1):
      if next_states[ps][2] != 0 :
        Max_Steps[ps] += 1  
    
    for ps in range(1, NumberOfPS + 1):
      if next_statesRandom[ps][2] != 0 :
        Max_StepsRandom[ps] += 1 
    for ps in range(1,NumberOfPS+1):
      Power_dict[f"{ps}"].append(actions[ps])
      AoI_dict[f"{ps}"].append(states[ps][1])
      Bits_Dict[f"{ps}"].append(states[ps][2])
      Power_dictRandom[f"{ps}"].append(actionsRandom[ps])
      AoI_dictRandom[f"{ps}"].append(statesRandom[ps][1])
      Bits_DictRandom[f"{ps}"].append(statesRandom[ps][2])
    states = next_states
    statesRandom = next_statesRandom

    total_timesteps += TimestepSize

  for i in range(1,SimulationParams.NumberOfPS+1):
    print(f"Average power usage for ps{i} with RL scheduler - Cooperative: {sum(Power_dict[f'{i}'])/len(Power_dict[f'{i}'])}")

  plt.figure(1)
  plt.title("AoI change with cooperative RL algorithm heuristic reward, Nₖ = 50")
  for i in range(1,SimulationParams.NumberOfPS+1):
    plt.plot(AoI_dict[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
  plt.axhline(y = SimulationParams.deadlines[0], color = "orange", linestyle = 'solid', label = "deadline", )
  plt.legend(loc = "best")

  plt.figure(3)
  plt.title("AoI change with random selection, Nₖ = 50")
  for i in range(1,SimulationParams.NumberOfPS+1):
    plt.plot(AoI_dictRandom[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
  plt.axhline(y = SimulationParams.deadlines[0], color = "orange", linestyle = 'solid', label = "deadline", )
  plt.legend(loc = "best")
  plt.figure(4)
  plt.title("Power consumption with cooperative RL algorithm heuristic reward, Nₖ = 50")
  for i in range(1,SimulationParams.NumberOfPS+1):
    plt.plot(Power_dict[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
  plt.legend(loc = "best")
  plt.show()

else:
  minAoiProb = np.float32("+inf")
  MinAvgAoIViolation = np.float32("+inf")
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
          res, signal, score , Avg_AoI , Avg_power, Avg_AoIViolation= evaluate_policy(Env= Env, policy= policy)
          MinAvgAoIViolation = min(Avg_AoIViolation)
          if signal == True:
            break_point += 1
          if break_point == 5:
            break
          f = open("./results/MinViolationProb.txt","w")
          f.write(str(minAoiProb))
          f.close()
          Jsonviolation = json.dumps(Avg_AoIViolation)
          f = open("./results/AoIVilationProbabilites.txt","w")
          f.write(Jsonviolation)
          f.close()
          for ps in range(1,NumberOfPS+1):
            Results[ps].append(res[ps])
          Json = json.dumps(Results)
          f = open("./results/Rewards.json","w")
          f.write(Json)
          f.close()
          for ps in range(1,NumberOfPS+1):
            Results2[ps].append(Avg_AoI[ps])
          Json2 = json.dumps(Results2)
          f = open("./results/AvgAoIs.json","w")
          f.write(Json2)
          f.close()
          for ps in range(1,NumberOfPS+1):
            Results3[ps].append(Avg_power[ps])
          Json3 = json.dumps(Results3)
          f = open("./results/AvgPowers.json","w")
          f.write(Json3)
          f.close()
    for ps in range(1,NumberOfPS+1):
      if total_timesteps%1 == 0 and total_timesteps > 200:
        policy[ps].train(1,replay_buffer[ps], batch_size, discount, tau, noise_clip, policy_freq)
    #   if total_timesteps%200 == 0: 
    #     print(actions)
      if total_timesteps%50 == 0 and total_timesteps > 200 and MinAvgAoIViolation < minAoiProb:
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
    if MinAvgAoIViolation < minAoiProb: 
      minAoiProb = MinAvgAoIViolation.copy() 
    next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, TimestepSize)  
    next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
    }
    # print(actions)
    # print(next_states)
    # T.sleep(1)
    for ps in range(1,NumberOfPS+1):
        TestAoIDict[ps].append(next_states[ps][1])
        TestTimeDict[ps].append(total_timesteps)

    for ps in range(1,NumberOfPS+1):
      episode_reward[ps] += rewards[ps]
      shaped_reward = 0 
      replay_buffer[ps].add((states[ps], actions[ps],next_states[ps], rewards[ps], terminal[ps], shaped_reward))
      episode_timesteps[ps] += 1
    timesteps_since_eval += TimestepSize
    states = next_states
    total_timesteps += TimestepSize
    epsilon = 5e-6
    if total_timesteps > start_timesteps:
       eps = eps - delta_eps if  eps > eps_min else eps_min
       expl_noise = expl_noise - epsilon if expl_noise > expl_noise_min else expl_noise_min
      #  print(expl_noise)
plt.figure(1)
plt.title("")
for ps in range(1,NumberOfPS+1):
    plt.plot( TestTimeDict[ps],TestAoIDict[ps])

plt.show()