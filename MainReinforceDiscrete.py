import copy
import itertools
from Utils import plotLearning
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
from REINFOCE.reinforce2 import PolicyGradientAgent
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

    while time < 1000:
      timesteps += TimestepSize
      for i in range(1,NumberOfPS+1):
        if timesteps > 1000: 
          timesteps = 0 
          states[ps] = Env.reset(ps= i)
          done[ps] = False
        action[i] = policy[i].choose_action(states[i])
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
      AoIErrorDict[i] = int((AoIErrorDict[i]/((1000*eval_episodes)/TimestepSize))*(100))
      PowerUsageDict[i] = ((PowerUsageDict[i]/((1000*eval_episodes)/TimestepSize)))

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
    
save_models = True
expl_noise_min = 0.01
epsilon = 0.9
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
max_timesteps = 10e6
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
Scores = dict()
new_policy = dict()
policyRandom = dict()
log_prob = dict()
Evaluations = []
Reward_Evaluations = []
score_history = dict()
replay_buffer = dict()
TestAoIDict = dict()
TestTimeDict = dict()
for ps in range(1, NumberOfPS + 1):
    TestAoIDict[ps] = []
    TestTimeDict[ps] = []

for i in range(1,NumberOfPS+1):
  states[i] = Env.reset(i)
  statesRandom[i] = EnvRandom.reset(i)
  policy[i] = PolicyGradientAgent(ALPHA=0.00004, input_dims=[SimulationParams.NumberOfTCh + 2], GAMMA=0.99,
                                n_actions=4, layer1_size= 128, layer2_size= 128)


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
    Scores[ps]  = 0
    score_history[ps] = []

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
        with open("./results/MinViolationProb.txt", "w") as f:
            f.write(str(minAoiProb))
        
        with open("./results/AoIVilationProbabilites.json", "w") as f:
            json.dump(Avg_AoIViolation, f)
        for ps in range(1,NumberOfPS+1):
            Results[ps].append(res[ps])
        Json = json.dumps(Results)
        f = open("./results/Rewards.json","w")
        f.write(Json)
        f.close()

        for ps in range(1, NumberOfPS + 1):
            Results2[ps].append(Avg_AoI[ps])
        with open("./results/AvgAoIs.json", "w") as f:
            json.dump(Results2, f)

        for ps in range(1, NumberOfPS + 1):
            Results3[ps].append((Avg_power[ps]))
        with open("./results/AvgPowers.json", "w") as f:
            json.dump(Results3, f)

    # Interaction with environment
    for ps in range(1, NumberOfPS + 1):
        if episode_timesteps[ps] >= 200:
            if total_timesteps != 0:
                policy[ps].learn()
                states[ps] = Env.reset(ps=ps)
                done[ps] = False
                episode_reward[ps] = []
                episode_timesteps[ps] = 0
                episode_num[ps] += 1
                score_history[ps].append(Scores[ps])

                Scores[ps] = 0
        
        if np.random.random() < epsilon:
            possible_actions = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
            actions[ps] = np.random.choice(possible_actions)     
        else:
            actions[ps] = policy[ps].choose_action(states[ps])

    if MinAvgAoIViolation < minAoiProb:
        minAoiProb = MinAvgAoIViolation


    for ps in range(1, NumberOfPS + 1):
        actions[ps] = np.array([float(actions[ps])])
    next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, TimestepSize)
    rewards = {ps: float(rewards[ps]) for ps in range(1, NumberOfPS + 1)}
    next_states = {
        i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
        for i in range(1, NumberOfPS + 1)
    }

    for ps in range(1, NumberOfPS + 1):
        TestAoIDict[ps].append(next_states[ps][1])
        TestTimeDict[ps].append(total_timesteps)
        policy[ps].store_rewards(rewards[ps])
        episode_reward[ps].append(rewards[ps])
        episode_timesteps[ps] += 1
        Scores[ps] += rewards[ps]
    states = next_states
    total_timesteps += TimestepSize
    timesteps_since_eval += TimestepSize
    if epsilon > 0 : 
        epsilon = epsilon - 1*5e-6
plotLearning(score_history, filename="S", window=25)