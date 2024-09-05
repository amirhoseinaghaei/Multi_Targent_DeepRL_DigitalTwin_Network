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
Results = {"15":{"40" : {"AoI":[], "Power":[]}, "30": {"AoI":[], "Power":[]}}, "20":{"40" : {"AoI":[], "Power":[]}, "45": {"AoI":[], "Power":[]}}}
bits = ["15"]
windows = ["30","40","45","50","60"]
pss = [5,6,7]
AoIViolationLit = []
gr = [2]
PowerList = []
if Test == True:
  for bit in bits:
    for w in windows:
      for p in pss:
       for g in gr:
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
        max_timesteps = 1e1
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
        print(p,2,p*[int(bit)])
        Env = CustumEnv(p, 2)
        # Env = CustumEnv(p, 2,p*[int(bit)], p*[100] , p*[int(w)], p*[2*int(w)])

        policy = dict()
        policy2 = dict()
        policy3 = dict()
        policy4 = dict()
        new_policy = dict()
        policyRandom = dict()
        Evaluations = []
        Reward_Evaluations = []
        replay_buffer = dict()

        for i in range(1,p+1):
          print(i)
          states[i] = Env.reset(i)
          policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= p, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
          policy[i].load(f"{i}th PS", f"./results/Final/{p}ps_{w}_{2*int(w)}_{bit}_{g}GR")
          print(f"{i}th PS", f"./results/Final/{p}ps_{w}_{2*int(w)}_{bit}_{g}GR")
          replay_buffer[i] = Replay_Buffer()

        central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= p, action_dim=  SimulationParams.NumberOfTCh)
        Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0 , 7:0} 

        max_Steps = 0
        AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[] , "7": []}
        Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7": []}
        Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7": []}

        while total_timesteps < max_timesteps:
          for ps in range(1,p+1):
            actions[ps]  = policy[ps].select_action(states[ps])
            actions[ps] = abs(actions[ps])

          next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, g)  
          # next_states, rewards, done, terminal = Env.step(actions, total_timesteps)    
          
          next_states = {
                  i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
                  for i in range(1, p + 1)        
          }
          for ps in range(1, p + 1):
            if next_states[ps][2] != 0 :
              Max_Steps[ps] += 1  
          
          for ps in range(1,p+1):
            Power_dict[f"{ps}"].append(actions[ps])
            AoI_dict[f"{ps}"].append(states[ps][1])
            Bits_Dict[f"{ps}"].append(states[ps][2])

        
          states = next_states

          total_timesteps += g
        average_power = 0 
        for ps in range(1, p + 1):
          print(sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}']))
          average_power += sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}'])
        average_power = average_power/p
        print(f"Average power usage of all PSs is {average_power}")
        PowerList.append(average_power)
        AoI_Violoation_Probability = {}
        for ps in range(1, p + 1):
          error = 0 
          for index in range(len(AoI_dict[f"{ps}"])):
            if index <= len(AoI_dict[f"{ps}"]) -2 :
                if AoI_dict[f"{ps}"][index] > 2*int(w):
                  error += 1
 
          AoI_Violoation_Probability[ps] = (error/len( AoI_dict[f"{ps}"]))*100
        mean = 0

        for ps in range(1, p + 1):
            mean += AoI_Violoation_Probability[ps]
        print(f"Average AoI violation probability of all PSs is {mean/p}")
        AoIViolationLit.append(mean/p)
        # Results[bit][rew]["AoI"].append(mean/p)
        # Results[bit][rew]["Power"].append(average_power)
  print(AoIViolationLit)
  print(PowerList)
plt.figure(1)

plt.title("Average AoI Violation, αᵢ = 0.1")
plt.plot([4,5,6], [0.42500000000000004, 9.41, 29.74166666666666] , label = "20 Mbits, window = 45ms, deadline = 90ms - αᵢ = 0.1",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [18.1375, 20.240000000000002, 35.21666666666665], label = "15 Mbits, window = 30ms, deadline = 60ms - αᵢ = 0.1",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [20, 4.8, 25.49166666666667] ,label = "15 Mbits, window = 40ms, deadline = 80ms - αᵢ = 0.1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper right")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(2)
plt.title("Average Power Usage, αᵢ = 0.1")
plt.plot([4,5,6], [0.16659538, 0.49,0.13158792] ,color = "blue", label = "Average Power Usage of all PSs, 20 Mbits, window = 45ms, deadline = 90ms - αᵢ = 0.1",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [0.41901, 0.28, 0.9989],color = "red", label = "Average Power Usage of all PSs, 15 Mbits, window = 30ms, deadline = 60ms - αᵢ = 0.1",  linestyle = "solid", marker = "*")
plt.plot([5,6,7], [0.30081, 0.25, 0.24868762] ,color = "green" ,label = "Average Power Usage of all PSs, 15 Mbits, window = 40ms, deadline = 80ms - αᵢ = 0.1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper right")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))


plt.figure(4)
plt.title("Convergence time VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [30300, 47400, 80900] , color = "blue",label = "W = 45, D = 90, B = 20 , GR = 5",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [20000, 60000, 83750], color = "red", label = "W = 45, D = 90, B = 20 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [62500, 166000, 180500] , color = "green", label = "W = 45, D = 90, B = 20 , GR = 1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper left")
plt.xlabel("Number of PS")
plt.ylabel("Convergence Time")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(5)
plt.title("optimal AoI Probability VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [49, 26, 25.83] , color = "blue",label = "W = 45, D = 90, B = 20 , GR = 5",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [ 4.2, 14.4, 20.16], color = "red",label = "W = 45, D = 90, B = 20 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [4.25, 6.5, 18.5] , color = "green",label = "W = 45, D = 90, B = 20 , GR = 1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper right")
plt.xlabel("Number of PS")
plt.ylabel("AoI Violation probability")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))


plt.figure(6)
plt.title("Convergence time VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [35200, 69900, 22000] ,color = "blue", label = "W = 40, D = 80, B = 15 , GR = 5",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [20750, 70000, 35500], color = "red", label = "W = 40, D = 80, B = 15 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [42000, 68500, 59000] , color = "green", label = "W = 40, D = 80, B = 15 , GR = 1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper left")
plt.xlabel("Number of PS")
plt.ylabel("Convergence Time")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(7)
plt.title("optimal AoI Probability VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [30.25, 5.8, 22] , color = "blue", label = "W = 40, D = 80, B = 15 , GR = 5",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [ 5, 7, 29], color = "red", label = "W = 40, D = 80, B = 15 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [2.25,3.2, 7.83] , color = "green", label = "W = 40, D = 80, B = 15 , GR = 1",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper right")
plt.xlabel("Number of PS")
plt.ylabel("AoI Violation probability")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))


plt.figure(8)
plt.title("Convergence time VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [75500, 38500, 201000] ,color = "green" , label = "W = 30, D = 60, B = 15 , GR = 1",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [50000, 40000, 57000], color = "red" , label = "W = 30, D = 60, B = 15 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [12000, 63800, 80000] ,color = "blue",  label = "W = 30, D = 60, B = 15 , GR = 5",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper left")
plt.xlabel("Number of PS")
plt.ylabel("Convergence Time")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(9)
plt.title("optimal AoI Probability VS Number of PS , αᵢ = 0.1")
plt.plot([4,5,6], [4.25, 12.6, 79.8] ,color = "green" , label = "W = 30, D = 60, B = 15 , GR = 1",  linestyle = "solid", marker = "o")
plt.plot([4,5,6], [ 8, 11.4, 27],color = "red", label = "W = 30, D = 60, B = 15 , GR = 2",  linestyle = "solid", marker = "*")
plt.plot([4,5,6], [22.25, 14.6, 28] ,color = "blue",  label = "W = 30, D = 60, B = 15 , GR = 5",  linestyle = "solid", marker = "^")
plt.legend(loc = "upper right")
plt.xlabel("Number of PS")
plt.ylabel("AoI Violation probability")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(3)
plt.title("Convergence time VS Time Granularity")
plt.plot([1,2,5], [59000, 35500, 22000], label = "6ps, W = 40, D = 80, B = 15",  linestyle = "solid", marker = "o")
plt.plot([1,2,5], [68500, 117500, 69900] , label = "5ps, W = 40, D = 80, B = 15",  linestyle = "solid", marker = "o")
plt.plot([1,2,5], [42000, 230000, 35200], label = "4ps, W = 40, D = 80, B = 15",  linestyle = "solid", marker = "o")
plt.plot([1,2,5], [201000, 57000, 80000], label = "6ps, W = 30, D = 60, B = 15",  linestyle = "solid", marker = "o")
plt.plot([1,2,5], [38500, 40000, 63800] , label = "5ps, W = 30, D = 60, B = 15",  linestyle = "solid", marker = "o")
plt.plot([1,2,5], [75500, 50000, 12000] , label = "4ps, W = 30, D = 60, B = 15",  linestyle = "solid", marker = "o")


plt.legend(loc = "upper left")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.show()

  # plt.plot(channelList, Results["120"]["nonsimple_0.01"]["AoI"] , label = "AoI Violation of all PSs, 120 bits - αᵢ = 0.1",  linestyle = "solid", marker = "*")
  # plt.plot(channelList, Results["140"]["nonsimple_0.01"]["AoI"] , label = "AoI Violation of all PSs, 140 bits - αᵢ = 0.1",  linestyle = "solid", marker = "*")

  # # plt.plot(channelList, Results["100"]["nonsimple_0.01"]["AoI"] , label = "AoI Violation of all PSs 100 bits- αᵢ using equation 21",  linestyle = "solid", marker = "s")
  # plt.ylabel("Average AoI Violation Probability")
  # plt.xlabel('Nₖ')
  # plt.xticks(channelList, labels=[f'{x}' for x in channelList])
  # plt.legend(loc = "upper left")

  # plt.figure(2)
  # plt.title("Power Usage Comparison with different number of channel states")
  # # plt.plot(channelList, Results["100"]["simple"]["Power"], label = "Average Power Usage of all PSs 100 bits - fixed αᵢ = 1",  linestyle = "dashed", marker = "*")
  # plt.plot(channelList, Results["100"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs, 100 bits - αᵢ = 0.1",  linestyle = "solid", marker = "*")
  # plt.plot(channelList, Results["120"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs, 120 bits - αᵢ = 0.1",  linestyle = "solid", marker = "*")
  # plt.plot(channelList, Results["140"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs, 140 bits - αᵢ = 0.1",  linestyle = "solid", marker = "*")

  # # plt.plot(channelList, Results["100"]["nonsimple_0.01"]["Power"] , label = "Average Power Usage of all PSs 100 bits - αᵢ using equation 21",  linestyle = "solid", marker = "s")
  # # plt.plot(channelList, Results["180"]["simple"]["Power"], label = "Average Power Usage of all PSs, 180 bits - αᵢ = 1",  linestyle = "dashed", marker = "*", color = "blue")
  # plt.ylabel("Average Power Usage")
  # plt.xlabel('Nₖ')
  # plt.xticks(channelList, labels=[f'{x}' for x in channelList])
  # plt.legend(loc = "upper left")
  # plt.show()

