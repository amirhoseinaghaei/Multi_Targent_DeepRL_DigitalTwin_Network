import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment2 import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
import time as T
from matplotlib import pyplot as plt
from MDDPG.Replay_Buffer  import Replay_Buffer
from MDDPG.Central_Critic import Central_Critic
from matplotlib import pyplot as plt



NumberOfPS = SimulationParams.NumberOfPS

def add_gaussian_noise(action, noise_std=0.1, min_action=0, max_action=1.0):
    noise = np.random.normal(loc=0.0, scale=noise_std, size=action.shape) if isinstance(action, np.ndarray) else np.random.normal(loc=0.0, scale=noise_std)
    noisy_action = action + noise
    clipped_action = np.clip(noisy_action, min_action, max_action)
    return clipped_action

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
    if AoIs[i] >= 100*80 - 1000:
        signal = False
        break
  return avg_reward, signal , scores, AoIs , powers

Test = True
Results = {"100":{"simple" : {"AoI":[], "Power":[]}, "nonsimple_0.01": {"AoI":[], "Power":[]}}, "120":{"simple" : {"AoI":[], "Power":[]}, "nonsimple_0.01": {"AoI":[], "Power":[]}}, "140":{"simple" : {"AoI":[], "Power":[]}, "nonsimple_0.01": {"AoI":[], "Power":[]}}}
bits = ["120"]
reward_mode = ["nonsimple_0.01"]
cpus = ["100","150","180"]
channelList = [20]
if Test == True:
  for bit in bits:
    for rew in reward_mode:
      for ch in channelList:
       for cpu in cpus:
        print(ch)
        save_models = True
        expl_noise_min = 0.001
        epsilon = 5e-5
        eps = 0.5
        delta_eps = 1e-5
        eps_min = 0.02
        expl_noise = 0.002
        eval_freq = 200
        batch_size = 100
        tau = 0.005
        discount = 0.99
        noise_clip =  0.005
        policy_noise = 0.005
        policy_freq = 2
        max_timesteps = 3e3
        start_timesteps = 0e3
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
        NumberOfBits = SimulationParams.NumberOfBits

        next_statesRandom = dict()
        Env = CustumEnv(NumberOfPS, ch,5*[int(bit)])
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
        print(f"5ps_{bit}_{ch}_withpower_{rew}")
        policy[1].load(f"{1}th PS", f"./pytorch_models/5ps_{bit}_{ch}_{cpu}cpu_withpower_{rew}")
        policy[2].load(f"{2}th PS", f"./pytorch_models/5ps_{bit}_{ch}_{cpu}cpu_withpower_{rew}")
        policy[3].load(f"{3}th PS", f"./pytorch_models/5ps_{bit}_{ch}_{cpu}cpu_withpower_{rew}")
        policy[4].load(f"{4}th PS", f"./pytorch_models/5ps_{bit}_{ch}_{cpu}cpu_withpower_{rew}")
        policy[5].load(f"{5}th PS", f"./pytorch_models/5ps_{bit}_{ch}_{cpu}cpu_withpower_{rew}")
        Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0} 


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
          # print(actions)


          for ps in range(1, NumberOfPS + 1):
            if next_states[ps][2] != 0 :
              Max_Steps[ps] += 1  
          
          for ps in range(1,NumberOfPS+1):
            Power_dict[f"{ps}"].append(actions[ps])
            AoI_dict[f"{ps}"].append(states[ps][1])
            Bits_Dict[f"{ps}"].append(states[ps][2])

        
          states = next_states

          total_timesteps += 1
        average_power = 0 
        for ps in range(1, NumberOfPS + 1):
          print(sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}']))
          average_power += sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}'])
        average_power = average_power/5
        print(f"Averagee power usage of all PSs is {average_power}")
        AoI_Violoation_Probability = {}
        for ps in range(1, NumberOfPS + 1):
          error = 0 
          for index in range(len(AoI_dict[f"{ps}"])):
            if index <= len(AoI_dict[f"{ps}"]) -2 :
              if int(bit) == 100:
                if AoI_dict[f"{ps}"][index] > 80:
                  error += 1
              if int(bit) == 120:
                if AoI_dict[f"{ps}"][index] > 90:
                  error += 1
              if int(bit) == 140:
                if AoI_dict[f"{ps}"][index] > 100:
                  error += 1
              if int(bit) == 180:
                if AoI_dict[f"{ps}"][index] > 120:
                  error += 1
          AoI_Violoation_Probability[ps] = (error/len( AoI_dict[f"{ps}"]))*100
        mean = 0

        for ps in range(1, NumberOfPS + 1):
            mean += AoI_Violoation_Probability[ps]
        print(f"Averagee AoI violation probability of all PSs is {mean/5}")
        Results[bit][rew]["AoI"].append(mean/5)
        Results[bit][rew]["Power"].append(average_power)
      
  plt.figure(1)
  plt.title("Comparison of AoI violation with different number of transmission bits  with simple and non-simple reward")
  # plt.plot(channelList, Results["100"]["simple"]["AoI"], label = "Average AoI Violation of all PSs 100 bits- fixed αᵢ = 1",  linestyle = "dashed", marker = "*")
#   plt.plot(channelList, Results["140"]["simple"]["AoI"], label = "Average AoI Violation of all PSs 120 bits- fixed αᵢ = 1",  linestyle = "dashed", marker = "*")
  plt.plot(cpus, Results["120"]["nonsimple_0.01"]["AoI"] , label = "AoI Violation of all PSs 120 bits - αᵢ using equation 21",  linestyle = "solid", marker = "*")
  # plt.plot(channelList, Results["100"]["nonsimple_0.01"]["AoI"] , label = "AoI Violation of all PSs 100 bits- αᵢ using equation 21",  linestyle = "solid", marker = "s")
  plt.ylabel("Average AoI Violation Probability")
  plt.xlabel('Nₖ')
  plt.xticks(cpus, labels=[f'{x}' for x in cpus])
  plt.legend(loc = "upper left")

  plt.figure(2)
  plt.title("Comparison Power Usage with different number of transmission bits  with simple and non-simple reward")
  # plt.plot(channelList, Results["100"]["simple"]["Power"], label = "Average Power Usage of all PSs 100 bits - fixed αᵢ = 1",  linestyle = "dashed", marker = "*")
  plt.plot(cpus, Results["120"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs 120 bits - αᵢ using equation 21",  linestyle = "solid", marker = "*")
  # plt.plot(channelList, Results["100"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs 100 bits - αᵢ using equation 21",  linestyle = "solid", marker = "s")
#   plt.plot(channelList, Results["120"]["simple"]["Power"], label = "Average Power Usage of all PSs 120 bits - fixed αᵢ = 1",  linestyle = "dashed", marker = "*")
  plt.ylabel("Average Power Usage Probability")
  plt.xlabel('Nₖ')
  plt.xticks(cpus, labels=[f'{x}' for x in cpus])
  plt.legend(loc = "upper left")
  plt.show()

