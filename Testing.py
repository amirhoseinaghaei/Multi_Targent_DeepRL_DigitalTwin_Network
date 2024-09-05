import numpy as np
from Environment import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
import time as T
from matplotlib import pyplot as plt
from MDDPG.Replay_Buffer  import Replay_Buffer
from MDDPG.Central_Critic import Central_Critic





Results = {"15":{"40" : {"AoI":[], "Power":[]}, "30": {"AoI":[], "Power":[]}}, "20":{"40" : {"AoI":[], "Power":[]}, "45": {"AoI":[], "Power":[]}}}
bits = ["15"]
windows = ["40"]
pss = [4]
AoIViolationLit = []
gr = [10]
PowerList = []
Test = True
if Test == True:
  for bit in bits:
    for w in windows:
      for nOfPs in pss:
       for g in gr:
        NumberOfPS = SimulationParams.NumberOfPS
        NumOfGainStates = SimulationParams.NumOfGainStates
        TimestepSize = SimulationParams.TimestepSize
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
        max_timesteps = 10e3
        start_timesteps = 00e3
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
        EnvRandom = CustumEnv(NumberOfPS, NumOfGainStates)
        Test = True
        policy = dict()
        policyRandom = dict()
        new_policy = dict()
        policyRandom = dict()
        Evaluations = []
        Reward_Evaluations = []
        replay_buffer = dict()
        TestAoIDict = dict()
        TestTimeDict = dict()
        next_statesRandom = dict()
        policy = dict()
        policy2 = dict()
        policy3 = dict()
        policy4 = dict()
        new_policy = dict()
        policyRandom = dict()
        Evaluations = []
        Reward_Evaluations = []
        replay_buffer = dict()
        Env = CustumEnv(nOfPs, NumOfGainStates)
        for i in range(1,nOfPs+1):
            states[i] = Env.reset(i)
            policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
            policy[i].load(f"{i}th PS", f"./results/Final/{nOfPs}ps_{w}_{2*int(w)}_{bit}_{g}GR")
            print(f"{i}th PS", f"./results/Final/{nOfPs}ps_{w}_{2*int(w)}_{bit}_{g}GR")
            replay_buffer[i] = Replay_Buffer()

        central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= nOfPs, action_dim=  SimulationParams.NumberOfTCh)
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
          for ps in range(1,nOfPs+1):
            actions[ps]  = policy[ps].select_action(states[ps])
            actions[ps] = abs(actions[ps])

          next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, g)            
          next_states = {
                  i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
                  for i in range(1, nOfPs + 1)        
          }
          for ps in range(1, nOfPs + 1):
            if next_states[ps][2] != 0 :
              Max_Steps[ps] += 1  
          
          for ps in range(1,nOfPs+1):
            Power_dict[f"{ps}"].append(actions[ps])
            AoI_dict[f"{ps}"].append(states[ps][1])
            Bits_Dict[f"{ps}"].append(states[ps][2])
          states = next_states
          total_timesteps += g
        average_power = 0 
        for ps in range(1, nOfPs + 1):
          print(sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}']))
          average_power += sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}'])
        average_power = average_power/nOfPs
        print(f"Average power usage of all PSs is {average_power}")
        PowerList.append(average_power)
        AoI_Violoation_Probability = {}
        for ps in range(1, nOfPs + 1):
          error = 0 
          errorCount = 0
          for index in range(len(AoI_dict[f"{ps}"])):
            if index <= len(AoI_dict[f"{ps}"]) -2 :
                if AoI_dict[f"{ps}"][index] > 2*int(w):
                  errorCount += 1
                  diff = (AoI_dict[f"{ps}"][index] - 2*int(w))/ (2*int(w))
                  error += diff 
          if errorCount != 0 :
            error = error/errorCount
 
          # AoI_Violoation_Probability[ps] = (error)*100
          AoI_Violoation_Probability[ps] = (errorCount/len( AoI_dict[f"{ps}"]))*100
        mean = 0

        print(AoI_Violoation_Probability)


        mean_value = np.mean(list(AoI_Violoation_Probability.values()))
        std_dev = np.std(list(AoI_Violoation_Probability.values()))
        cov = std_dev / mean_value
        maxAoI = max(AoI_Violoation_Probability.values())
        for ps in range(1, nOfPs + 1):
            mean += AoI_Violoation_Probability[ps]
        print(f"Average AoI violation probability of all PSs is {mean/nOfPs}")
        
        AoIViolationLit.append(mean/nOfPs)
  
  plt.figure(30)
  plt.plot(AoI_dict["1"])
  print(f"Max AoI is: {maxAoI}")
  print(f"CoV of AoI: {cov}")
  print(f"Std of AoI: {std_dev}")
  print(f"Average of AoI: {AoIViolationLit}")
  print(PowerList)

plt.figure(10)

plt.title("Average AoI Violation for 20 Mbits and αᵢ = 0.1")
plt.plot([30,40,45,50,60], [29, 3.46, 0.16, 0.12, 0.25] , label = "4PS",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([30,40,45,50,60], [40, 9.2, 1.63, 0.81, 0.45] , label = "5PS",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([30,40,45,50,60], [93, 14, 6.76, 4.48, 2.42] , label = "6PS",  linestyle = "solid", marker = "^", color = "green")
plt.plot([30,40,45,50,60], [70, 34.77, 17.29, 28.37, 1.28] , label = "7PS",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([30,40,45,50,60], [99, 99, 99.35, 95, 98] , label = "8PS",  linestyle = "solid", marker = "<", color = "m")
plt.xlabel("Window Size")
plt.ylabel("Probability of AoI Violation")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.savefig('AoIComparison vs W.eps', format='eps')



# plt.figure(16)

# plt.title("Average AoI Violation for 15 Mbits, 5PS,40ms window, αᵢ = 0.1")
# plt.plot([50,40,30,20,10,5], [0.01,0.096, 1.49, 5.6, 48, 79] ,  linestyle = "solid", marker = "o", color = "blue")
# plt.xlabel("CPU computation power(cycle per milisec)")
# plt.ylabel("Probability of AoI Violation")
# plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))





plt.figure(16)

plt.title("Average AoI Violation Rate, $B_{max}$ = 15 Mbits, αᵢ = 0.1, $D_{max}$ = 80ms")
plt.plot([50,40,30,20,10,5], [0.01,0.096, 1.49, 5.6, 48, 79] ,  linestyle = "solid", marker = "o", color = "blue")
plt.xlabel("$f_{c}$ (M cycles / ms)")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))



plt.figure(17)

plt.title("Average AoI Violation Rate, $D_{max}$ = 80ms, αᵢ = 0.1")
plt.plot([15,20,25,30], [5.16, 5.96, 33, 38] , label = "5PS",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([15,20,25,30], [0.9,0.5,5.97,18] , label = "4PS",  linestyle = "solid", marker = "*", color = "orange")
plt.xlabel("$B_{max}$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")


plt.figure(15)

plt.title("Average AoI Violation Rate, $B_{max}$ = 15 Mbits, αᵢ = 0.1")
plt.plot([60,70,80,90,100,110,120], [30,0.13, 7.91, 1.89, 0.35, 0.2,0.25] , label = "4PS",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([60,70,80,90,100,110,120], [33,6.26 ,13, 10, 0.43, 0.48, 0.18] , label = "5PS",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([60,70,80,90,100,110,120], [34,21 ,26, 5.8, 0.18, 1.12, 3.2] , label = "6PS",  linestyle = "solid", marker = "^", color = "green")
plt.plot([60,70,80,90,100,110,120], [51, 44,29,25.9, 14, 7.5, 1.28] , label = "7PS",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([60,70,80,90,100,110,120], [60, 65,34, 34, 26,44, 99] , label = "8PS",  linestyle = "solid", marker = "<", color = "m")
plt.xlabel("$D_{max}$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))


plt.figure(18)
plt.title("Avg AoI Violation Rate Comparison, $D_{max}$ = 80ms, $B_{max}$ = 15Mbits, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.38,5.2,24,36,41,94 ] , label = "Centralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "m")
# plt.plot([4,5,6,7,8,9], [8,15,48,93,99,99 ] , label = "Centralized Critic Multi Agent Rl - Time Granularity = 10",  linestyle = "solid", marker = "o", color = "orange")
plt.plot([4,5,6,7,8,9], [ 0.14,2.6,15,31,99,99] , label = "Decentralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [66,74,94,99,99,99 ] , label = "Round Robin",  linestyle = "solid", marker = "s", color = "green")
plt.plot([4,5,6,7,8,9], [48.64,58,66,97,98,98] , label = "Cooperative Game Theory",  linestyle = "solid", marker = "s", color = "aqua")

plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(19)
plt.title("Avg Power Usage Comparison, $D_{max}$ = 80ms, $B_{max}$ = 15Mbits, αᵢ = 0.1")

plt.plot([4,5,6,7,8,9], [0.3,0.29,0.08,0.17,0.25,0.72] , label = "Centralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "m")
# plt.plot([4,5,6,7,8,9], [0.26,0.47,0.54,0.9,1,1] , label = "Centralized Critic Multi Agent Rl - Time Granularity = 10",  linestyle = "solid", marker = "o", color = "orange")
plt.plot([4,5,6,7,8,9], [ 0.3,0.2,0.23,0.3,0.97,1]  , label = "Decentralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [1,1,1,1,1,1] , label = "Round Robin",  linestyle = "solid", marker = "s", color = "green")
plt.plot([4,5,6,7,8,9], [0.25,0.2,0.16,0.14,0.125,0.11], label = "Cooperative Game Theory",  linestyle = "solid", marker = "s", color = "aqua")

plt.xlabel("$N$")
plt.ylabel("Average Power Usage")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(28)
plt.title("Average AoI Violation Comparison, 45ms- 15Mbits , αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.39,4,1.4,35,75,98 ] , label = "Centralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.13,0.27,6.7,20,40,99] , label = "Decentralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [63,71,89,99,99,99 ] , label = "Round Robin",  linestyle = "solid", marker = "s", color = "green")
plt.plot([4,5,6,7,8,9], [47,58,66,95,97,97] , label = "Cooperative Game Theory",  linestyle = "solid", marker = "s", color = "aqua")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(29)
plt.title("Average Power Usage Comparison, 45ms- 15Mbits , αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.26,0.21,0.26,0.21,0.72,0.95] , label = "Centralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [ 0.19,0.29,0.06,0.3,0.22,1] , label = "Decentralized Critic Multi Agent Rl",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [1,1,1,1,1,1] , label = "Round Robin",  linestyle = "solid", marker = "s", color = "green")
plt.plot([4,5,6,7,8,9], [0.25,0.2,0.16,0.14,0.125,0.11], label = "Cooperative Game Theory",  linestyle = "solid", marker = "s", color = "aqua")
plt.axhline(y = 0.25, color = "red", linestyle = 'solid', label = "Maximum Acceptable Average Power", )

plt.xlabel("Number of PS")
plt.ylabel("Average Power Usage")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(1)

plt.title("Average AoI Violation Rate, $B_{max}$ = 20 Mbits, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [29, 40, 93, 70, 99, 99] , label = "$D_{max}$ = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [3.46,9.2,14,34.77,99,99] , label = "$D_{max}$ = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.16,1.63,6.76,17.29,99.35,99.39] , label = "$D_{max}$ = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.12,0.81,4.48,28.37,95,99] , label = "$D_{max}$ = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.25, 0.45, 2.42,1.28,98,99] , label = "$D_{max}$ = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
# plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(2)
plt.title("Average Power Usage, $B_{max}$ = 20 Mbits, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.08, 0.18, 0.94, 0.9, 0.99, 1] , label = "$D_{max}$ = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [0.09,0.16,0.12,0.09,0.96,0.99] , label = "$D_{max}$ = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.16,0.25,0.21,0.23,0.93,0.97] , label = "$D_{max}$ = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.09,0.2,0.13,0.13,0.88,1] , label = "$D_{max}$ = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.21, 0.25,0.12,0.14,0.98,0.99] , label = "$D_{max}$ = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.legend(loc = "best")
plt.xlabel("$N$")
plt.ylabel("Average Power Usage")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(3)

plt.title("Average AoI Violation Rate, $B_{max}$ = 15 Mbits, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [30, 33, 34, 51, 60, 77] , label = "$D_{max}$ = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [7.91,13,26,29,34,61] , label = "$D_{max}$ = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [1.88,10,5.8,25.9,34,99] , label = "$D_{max}$ = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.35,0.43,0.18,14,26,99] , label = "$D_{max}$ = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.29, 0.18, 3.2,0.11,99,99] , label = "$D_{max}$ = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.legend(loc = "best")
plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(31)

plt.title("Average AoI Violation Percetnage Above Deadline for 15 Mbits and αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [10.768398268398267,  10.768398268398267, 66.71985297850345, 8112.629105969965, 8082.903741704923, 8200] , label = "window = 30ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [9.143907563025211,15.770518984725005,52.84366524871539,140.13885524283336,7128.571428571429,7077.142857142858] , label = "window = 35ms",  linestyle = "solid", marker = "*", color = "black")
plt.plot([4,5,6,7,8,9], [8.44,22.3,59,110,144,5822.306930005018] , label = "window = 40ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [7.5,22,14,151.52063424592365,5319.826436795995,5441.722070379657] , label = "window = 45ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [3.5,6.2,9,40,255,4985.111111111111] , label = "window = 50ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [1.36,6.5,17.75,14.9,81,4570] , label = "window = 55ms",  linestyle = "solid", marker = "s", color = "c")
plt.plot([4,5,6,7,8,9], [1, 5.6, 12.01,5.9, 4150,4165.833333333333] , label = "window = 60ms",  linestyle = "solid", marker = "<", color = "m")
plt.legend(loc = "best")
plt.xlabel("Number of PSs")
plt.ylabel("Average AoI Violation Percetnage")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(4)
plt.title("Average Power Usage, $B_{max}$ = 15 Mbits, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.09, 0.15, 0.06, 0.15, 0.044, 0.1] , label = "$D_{max}$ = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [0.23,0.28,0.14,0.10,0.17,0.16] , label = "$D_{max}$ = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.19,0.17,0.21,0.2,0.32,0.99] , label = "$D_{max}$ = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.08,0.14,0.23,0.1,0.2,0.99] , label = "$D_{max}$ = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.11, 0.3,0.21,0.20,0.99,0.96] , label = "$D_{max}$ = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.legend(loc = "best")
plt.xlabel("$N$")
plt.ylabel("Average Power Usage")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(5)
plt.title("Average AoI Violation Rate Comparison, $D_{max}$ = 60ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [30, 33, 34, 51, 60, 77] , label = "$B_{max}$ = 15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [29, 40, 93, 70, 99, 99] , label = "$B_{max}$ = 20 Mbits",  linestyle = "solid", marker = "s", color = "green")

plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(6)
plt.title("Average AoI Violation Rate Comparison, $D_{max}$ = 80ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [7.91,13,20,29,34,61] , label = "$B_{max}$ = 15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [3.46,9.2,14,34.77,99,99] , label = "$B_{max}$ = 20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(7)
plt.title("Average AoI Violation Rate Comparison, $D_{max}$ = 90ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [1.88,10,5.8,25.9,34,99] , label = "$B_{max}$ = 15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.16,1.63,6.76,17.29,99.35,99.39] , label = "$B_{max}$ = 20 Mbits",  linestyle = "solid", marker = "s", color = "green")

plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(8)
plt.title("Average AoI Violation Rate Comparison, $D_{max}$ = 100ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.35,0.43,0.18,14,26,99] , label = "$B_{max}$ = 15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.12,0.81,4.48,28.37,95,99] , label = "$B_{max}$ = 20 Mbits",  linestyle = "solid", marker = "s", color = "green")

plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(9)
plt.title("Average AoI Violation Rate Comparison, $D_{max}$ = 120ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.29, 0.18, 3.2,0.11,99,99] , label = "$B_{max}$ = 15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.25, 0.45, 2.42,1.28,98,99], label = "$B_{max}$ = 20 Mbits",  linestyle = "solid", marker = "s", color = "green")

plt.xlabel("$N$")
plt.ylabel("AoI Violation Rate")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.show()


# for ps in range(1, NumberOfPS + 1):
#     TestAoIDict[ps] = []
#     TestTimeDict[ps] = []

# for i in range(1,NumberOfPS+1):
#   states[i] = Env.reset(i)
#   statesRandom[i] = EnvRandom.reset(i)
#   policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
#   replay_buffer[i] = Replay_Buffer()

# central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= NumberOfPS, action_dim=  SimulationParams.NumberOfTCh)



# for i in range(1,SimulationParams.NumberOfPS+1):
#     policy[i].load(f"{i}th PS", f"./results/Final/{SimulationParams.NumberOfPS}ps_{SimulationParams.windows[0]}_{SimulationParams.windows[0]*2}_{SimulationParams.NumberOfBits[0]}_{TimestepSize}GR")


# Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0,7:0, 8:0,9:0} 
# Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0,7:0, 8:0,9:0} 


# max_Steps = 0
# AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
# Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
# Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
# AoI_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[],"7":[], "8":[], "9":[]}
# Power_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
# Bits_DictRandom = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}


# while total_timesteps < max_timesteps:
#     for ps in range(1,NumberOfPS+1):
#         actions[ps]  = policy[ps].select_action(states[ps])
#         actions[ps] = abs(actions[ps])
#         if statesRandom[ps][2] <= 100:
#             actionsRandom[ps] = np.array([0])
#         else:
#             actionsRandom[ps] = np.array([np.random.uniform(0,1)]) 
# # next_states, rewards, done, terminal = Env.step(actions, total_timesteps)  
# next_states, rewards, done, terminal = Env.stepWithStepSize(actions, total_timesteps, TimestepSize)  

# next_statesRandom, rewardsRandom, doneRandom, terminalRandom = EnvRandom.stepWithStepSize(actionsRandom, total_timesteps, TimestepSize)     
# next_states = {
#         i: np.array(list(next_states[0][f"ps{i}"]) + [next_states[1][f"ps{i}"]] + [next_states[2][f"ps{i}"]])
#         for i in range(1, NumberOfPS + 1)          
# }
# next_statesRandom = {
#         i: np.array(list(next_statesRandom[0][f"ps{i}"]) + [next_statesRandom[1][f"ps{i}"]] + [next_statesRandom[2][f"ps{i}"]])
#         for i in range(1, NumberOfPS + 1)
        
# }
# for ps in range(1, NumberOfPS + 1):
#     if next_states[ps][2] != 0 :
#         Max_Steps[ps] += 1  

# for ps in range(1, NumberOfPS + 1):
#     if next_statesRandom[ps][2] != 0 :
#         Max_StepsRandom[ps] += 1 
# for ps in range(1,NumberOfPS+1):
#     Power_dict[f"{ps}"].append(actions[ps])
#     AoI_dict[f"{ps}"].append(states[ps][1])
#     Bits_Dict[f"{ps}"].append(states[ps][2])
#     Power_dictRandom[f"{ps}"].append(actionsRandom[ps])
#     AoI_dictRandom[f"{ps}"].append(statesRandom[ps][1])
#     Bits_DictRandom[f"{ps}"].append(statesRandom[ps][2])
# states = next_states
# statesRandom = next_statesRandom

# total_timesteps += TimestepSize

# for i in range(1,SimulationParams.NumberOfPS+1):
#     print(f"Average power usage for ps{i} with RL scheduler - Cooperative: {sum(Power_dict[f'{i}'])/len(Power_dict[f'{i}'])}")

# plt.figure(1)
# plt.title("AoI change with cooperative RL algorithm heuristic reward, Nₖ = 50")
# for i in range(1,SimulationParams.NumberOfPS+1):
#     plt.plot(AoI_dict[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
# plt.axhline(y = SimulationParams.deadlines[0], color = "orange", linestyle = 'solid', label = "deadline", )
# plt.legend(loc = "best")

# plt.figure(3)
# plt.title("AoI change with random selection, Nₖ = 50")
# for i in range(1,SimulationParams.NumberOfPS+1):
#     plt.plot(AoI_dictRandom[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
# plt.axhline(y = SimulationParams.deadlines[0], color = "orange", linestyle = 'solid', label = "deadline", )
# plt.legend(loc = "best")
# plt.figure(4)
# plt.title("Power consumption with cooperative RL algorithm heuristic reward, Nₖ = 50")
# for i in range(1,SimulationParams.NumberOfPS+1):
#     plt.plot(Power_dict[f"{i}"] , label = f"ps{i}",  linestyle = "dashed")
# plt.legend(loc = "best")
# plt.show()
