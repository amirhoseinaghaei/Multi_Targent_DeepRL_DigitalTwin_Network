import copy
import itertools
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
from REINFOCE.reinforce import REINFORCE
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
        action[i], _ = policy[i].select_action(states[i])
        action[i] = abs(np.array([action[i]]))
        
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
eval_freq = 5000
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
log_probs = dict()
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
action_tensor = dict()
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
log_prob = dict()
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
  policy[i] = REINFORCE(state_dim= SimulationParams.NumberOfTCh + 2, action_dim= SimulationParams.NumberOfTCh)

if Test == True:

    print("tews")
else:


    # Initialize parameters
    minAoiProb = np.float32("+inf")
    MinAvgAoIViolation = np.float32("+inf")
    break_point = 0
    Results = {}
    Results2 = {}
    Results3 = {}
    TimeSteps = {"Time": 0}

    for ps in range(1, NumberOfPS + 1):
        Results[ps] = []
        Results2[ps] = []
        Results3[ps] = []

    # Create directories
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Initialize environment variables
    done = {ps: False for ps in range(1, NumberOfPS + 1)}
    episode_num = {ps: 0 for ps in range(1, NumberOfPS + 1)}
    episode_reward = {ps: [] for ps in range(1, NumberOfPS + 1)}
    log_probs = {ps: [] for ps in range(1, NumberOfPS + 1)}
    log_prob = {ps: 0 for ps in range(1, NumberOfPS + 1)}
    max_episode_reward = {ps: -np.inf for ps in range(1, NumberOfPS + 1)}
    min_episode_reward = {ps: np.inf for ps in range(1, NumberOfPS + 1)}
    episode_timesteps = {ps: 0 for ps in range(1, NumberOfPS + 1)}

    total_timesteps = 0
    timesteps_since_eval = 0
    signal = False

    # Setup plotting
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
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        plt.draw()

    # Main training loop
    while total_timesteps < max_timesteps:
        if timesteps_since_eval >= eval_freq:
            # Evaluate the policy
            TimeSteps["Time"] = total_timesteps
            with open("./results/Time.json", "w") as f:
                json.dump(TimeSteps, f)

            timesteps_since_eval %= eval_freq
            res, signal, score, Avg_AoI, Avg_power, Avg_AoIViolation = evaluate_policy(Env=Env, policy=policy)
            MinAvgAoIViolation = min(Avg_AoIViolation)
            
            if signal:
                break_point += 1
            if break_point == 5:
                break
            
            with open("./results/MinViolationProb.txt", "w") as f:
                f.write(str(minAoiProb))
            
            with open("./results/AoIVilationProbabilites.json", "w") as f:
                json.dump(Avg_AoIViolation, f)


            for ps in range(1,NumberOfPS+1):
                Results[ps].append(float(res[ps][0]))
            Json = json.dumps(Results)
            f = open("./results/Rewards.json","w")
            f.write(Json)
            f.close()

            for ps in range(1, NumberOfPS + 1):
                Results2[ps].append(Avg_AoI[ps])
            with open("./results/AvgAoIs.json", "w") as f:
                json.dump(Results2, f)

            for ps in range(1, NumberOfPS + 1):
                Results3[ps].append(float(Avg_power[ps][0]))
            with open("./results/AvgPowers.json", "w") as f:
                json.dump(Results3, f)

        # Interaction with environment
        for ps in range(1, NumberOfPS + 1):
            if episode_timesteps[ps] >= 1000:
                if total_timesteps != 0:
                    returns = policy[ps].compute_returns(episode_reward[ps])
                    policy[ps].update_policy(log_probs[ps], returns)
                    states[ps] = Env.reset(ps=ps)
                    done[ps] = False
                    log_probs[ps] = []
                    episode_reward[ps] = []
                    episode_timesteps[ps] = 0
                    episode_num[ps] += 1

            actions[ps], action_tensor[ps] = policy[ps].select_action(states[ps])
            
        if MinAvgAoIViolation < minAoiProb:
            minAoiProb = MinAvgAoIViolation
        
        for ps in range(1, NumberOfPS + 1):
            actions[ps] = np.array([actions[ps]])
        next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, TimestepSize)
        # print(f"State : {states}")
        # print(f"Action : {actions}")        
        # print(f"Reward : {states}")        
        rewards = {ps: float(rewards[ps]) for ps in range(1, NumberOfPS + 1)}
        next_states = {
            i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
            for i in range(1, NumberOfPS + 1)
        }
        # print(f"Next State : {next_states}")        

        for ps in range(1, NumberOfPS + 1):
            TestAoIDict[ps].append(next_states[ps][1])
            TestTimeDict[ps].append(total_timesteps)
            episode_reward[ps].append(rewards[ps])
            log_probs[ps].append(action_tensor[ps])
            episode_timesteps[ps] += 1
            
        states = next_states
        total_timesteps += TimestepSize
        timesteps_since_eval += TimestepSize
       