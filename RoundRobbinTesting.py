import copy
import datetime
import itertools
import numpy as np
from Environment import CustumEnv , SimulationParams
from MDDPG.MTD3 import NewTD3 
import time as T
from matplotlib import pyplot as plt
from MDDPG.Replay_Buffer  import Replay_Buffer
from MDDPG.Central_Critic import Central_Critic
from datetime import datetime





# Shapley value calculation


Results = {"15":{"40" : {"AoI":[], "Power":[]}, "30": {"AoI":[], "Power":[]}}, "20":{"40" : {"AoI":[], "Power":[]}, "45": {"AoI":[], "Power":[]}}}
bits = ["15"]
windows = ["40"]
pss = [6]
AoIViolationLit = []
AoIViolationLitRound = []
AoIViolationLitGameTheory = []
gr = [2]
PowerList = []
PowerListRound = []
PowerListGameTheory = []
Test = True
if Test == True:
  for bit in bits:
    for w in windows:
      for nOfPs in pss:
       for g in gr:
        RoundPS =1
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
        max_timesteps = 100e2
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
        terminalRound = dict()
        terminalGameTheory = dict()
        states = dict()
        statesRound = dict()
        statesGameTheory = dict()
        actions = dict()
        actionsRoundRobbin = dict()
        actionsGameTheory = dict()
        next_states = dict()
        next_statesRound = dict()
        next_statesGameTheory = dict()
        actionsRandom = dict()
        statesRandom = dict()
        rewardsRandom = dict()
        rewardRound = dict()
        rewardGameTheory = dict()
        doneRandom = dict()
        doneRoundRobbin = dict()
        doneGameTheory = dict()
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
        policyRoundRobbin = dict()
        policyGameTheory = dict()
        Evaluations = []
        Reward_Evaluations = []
        replay_buffer = dict()
        replay_bufferRoundRobbin = dict()
        replay_bufferGameTheory = dict()
        Env = CustumEnv(nOfPs, NumOfGainStates)
        RoundRobbinEnv = CustumEnv(nOfPs, NumOfGainStates)
        GameTheoryEnv = CustumEnv(nOfPs, NumOfGainStates)
        for i in range(1,nOfPs+1):
            
            actionsGameTheory[i] = np.array([0.5])
            states[i] = Env.reset(i)
            statesGameTheory[i] = GameTheoryEnv.reset(i)
            statesRound[i] = RoundRobbinEnv.reset(i)
            policy[i] = NewTD3(state_dim= SimulationParams.NumberOfTCh + 2, N= nOfPs, epsilon= epsilon, action_dim= SimulationParams.NumberOfTCh, max_action= 1)
            policy[i].load(f"{i}th PS", f"./results/Final/{nOfPs}ps_{w}_{2*int(w)}_{bit}_{g}GR")
            print(f"{i}th PS", f"./results/Final/{nOfPs}ps_{w}_{2*int(w)}_{bit}_{g}GR")
            replay_buffer[i] = Replay_Buffer()

        central_critic = Central_Critic(state_dim= SimulationParams.NumberOfTCh + 2, N= nOfPs, action_dim=  SimulationParams.NumberOfTCh)
        Max_Steps = {1:0, 2:0, 3:0, 4:0 ,5:0, 6:0,7:0, 8:0,9:0} 
        Max_StepsRandom = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0,7:0, 8:0,9:0} 
        Max_StepsRound = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0,7:0, 8:0,9:0} 
        Max_StepsGameTheory = {1:0, 2:0, 3:0 , 4:0 , 5:0, 6:0,7:0, 8:0,9:0} 
        max_Steps = 0
        AoI_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        Power_dict = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        Bits_Dict = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        AoI_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[],"7":[], "8":[], "9":[]}
        Power_dictRandom = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        Bits_DictRandom = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        AoI_dictRound = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[],"7":[], "8":[], "9":[]}
        Power_dictRound = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        Bits_DictRound = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        AoI_dictGameTheory = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[],"7":[], "8":[], "9":[]}
        Power_dictGameTheory = {"1":[] , "2":[], "3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}
        Bits_DictGameTheory = {"1":[] , "2":[],"3":[], "4":[], "5":[], "6":[], "7":[], "8":[], "9":[]}

        while total_timesteps < max_timesteps:

          actionsGameTheory1 = copy.deepcopy(actionsGameTheory)
       
          def shapley_value(num_ps, P_max=1.0):
                shapley_vals = np.zeros(num_ps)
                all_ps = list(range(num_ps))
                for i in range(num_ps):
                    for S in itertools.combinations(all_ps, nOfPs):
                        if i in S:
                            # S without i
                            S_wo_i = list(S)
                            S_wo_i.remove(i)
                            P_S = np.array([P_max if j in S else 0 for j in range(num_ps)])
                            P_S_wo_i = np.array([P_max if j in S_wo_i else 0 for j in range(num_ps)])
                            
                            P_S = {i + 1: np.array([P_S[i]]) for i in range(len(P_S))}
                            P_S_wo_i = {i + 1: np.array([P_S_wo_i[i]]) for i in range(len(P_S_wo_i))}
                            utilitySum = 0 
                            WutilitySum = 0
                            copyP_S = copy.deepcopy(P_S)
                            copyP_S_wo_i = copy.deepcopy(P_S_wo_i)
                            for ps in range(1,nOfPs+1):
                                copyP_S.pop(ps)
                                copyP_S_wo_i.pop(ps)
                                utility = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate2(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
                                                                            GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
                                                                            copyP_S,
                                                                            P_S[ps])
                                wutility = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate2(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
                                                                            GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
                                                                            copyP_S_wo_i,
                                                                            P_S_wo_i[ps])
                                WutilitySum += wutility
                                utilitySum += utility                            
                            marginal_contribution = utilitySum - WutilitySum
                            shapley_vals[i] += marginal_contribution / (len(S) * (num_ps - len(S) + 1))
                
                return shapley_vals

        #   def shapley_value(num_ps, P_max=1.0, num_samples=1000):
        #             shapley_vals = np.zeros(num_ps)
        #             all_ps = list(range(num_ps))
        #             num_coalitions = 100  # Limit the number of samples

        #             for i in range(num_ps):
        #                 sampled_coalitions = itertools.islice(itertools.combinations(all_ps, num_ps), num_coalitions)
        #                 start = (datetime.now())
        #                 for S in sampled_coalitions:
        #                     if i in S:
        #                         # S without i
        #                         S_wo_i = list(S)
        #                         S_wo_i.remove(i)
        #                         P_S = {j + 1: np.array([P_max if j in S else 0]) for j in range(num_ps)}
        #                         P_S_wo_i = {j + 1: np.array([P_max if j in S_wo_i else 0]) for j in range(num_ps)}
                                
        #                         utility_sum = 0
        #                         wutility_sum = 0
        #                         for ps in range(1, num_ps + 1):
        #                             copyP_S = copy.deepcopy(P_S)
        #                             copyP_S_wo_i = copy.deepcopy(P_S_wo_i)
        #                             copyP_S.pop(ps)
        #                             copyP_S_wo_i.pop(ps)

        #                             utility = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate(
        #                                 GameTheoryEnv._get_obs(ps)["Channel_Gain"],
        #                                 GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
        #                                 copyP_S,
        #                                 P_S[ps])
                                    
        #                             wutility = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate(
        #                                 GameTheoryEnv._get_obs(ps)["Channel_Gain"],
        #                                 GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
        #                                 copyP_S_wo_i,
        #                                 P_S_wo_i[ps])
                                    
        #                             utility_sum += utility
        #                             wutility_sum += wutility

        #                         marginal_contribution = utility_sum - wutility_sum
        #                         shapley_vals[i] += marginal_contribution / (len(S) * (num_ps - len(S) + 1))

        #             # Average the values over the number of samples
        #             shapley_vals /= num_coalitions
        #             return shapley_vals

          SHV = shapley_value(num_ps= nOfPs)
          for ps in range(1,nOfPs+1):
            if np.min(SHV) < 0:
                adjusted_shapley_vals = SHV - np.min(SHV)
            else:
                adjusted_shapley_vals = SHV
            # if sum(adjusted_shapley_vals) == 0 :
            #     actionsGameTheory[ps] = np.array([(1.0/nOfPs)]) 
            # else:
            actionsGameTheory[ps] = np.array([float((adjusted_shapley_vals[ps-1]/sum(adjusted_shapley_vals))*1.0)])
            rest_action = copy.deepcopy(actionsGameTheory1)
            rest_action.pop(ps)
            bestUtility = 0
            # for P in np.arange(0, 1 + 0.01, 0.01):what 
            #     utility = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
            #                                                                GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
            #                                                                rest_action,
            #                                                                np.array([P]))
                
            #     if  utility - bestUtility > 1 :
            #        best_utility = utility
            #        actionsGameTheory[ps] = np.array([P])
            # PowerRate = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate2(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
            #                                                                GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
            #                                                                rest_action,
            #                                                                actionsGameTheory1[ps])

            # increasedPowerRate = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate2(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
            #                                                                GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
            #                                                                rest_action,
            #                                                                actionsGameTheory1[ps] + 0.01)
            # decreasedPowerRate = GameTheoryEnv.Wireless_Tr_Channels.calculate_transmission_rate2(GameTheoryEnv._get_obs(ps)["Channel_Gain"],
            #                                                                GameTheoryEnv._get_info(ps)["Interference_Channel_Gain"],
            #                                                                rest_action,
            #                                                                actionsGameTheory1[ps] - 0.01)
            # if increasedPowerRate > PowerRate + 0.001 : 
            #     actionsGameTheory[ps] = (min(actionsGameTheory[ps][0] + 0.01,1))
            #     actionsGameTheory[ps] = np.array([actionsGameTheory[ps]])
            # elif decreasedPowerRate > PowerRate + 0.001:
            #     actionsGameTheory[ps] = (max(actionsGameTheory[ps][0] - 0.01,0))
            #     actionsGameTheory[ps] = np.array([actionsGameTheory[ps]])
            actions[ps]  = policy[ps].select_action(states[ps])
            actions[ps] = abs(actions[ps])
            if RoundPS == ps:
              actionsRoundRobbin[ps] = np.array([1])
            else:
              actionsRoundRobbin[ps] = np.array([0])
            RoundPS += 1
            if RoundPS > NumberOfPS:
                RoundPS = RoundPS%NumberOfPS
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
          next_statesRound, rewardRound, doneRoundRobbin, terminalRound = RoundRobbinEnv.stepWithStepSize(actionsRoundRobbin, total_timesteps, g)
          next_statesRound = {
                  i: np.array(list(next_statesRound[0][f"ps{i}"]) + [next_statesRound[1][f"ps{i}"]] + [next_statesRound[2][f"ps{i}"]])
                  for i in range(1, nOfPs + 1)        
          }
          for ps in range(1, nOfPs + 1):
            if next_statesRound[ps][2] != 0 :
              Max_StepsRound[ps] += 1  
          for ps in range(1,nOfPs+1):
            Power_dictRound[f"{ps}"].append(actionsRoundRobbin[ps])
            AoI_dictRound[f"{ps}"].append(statesRound[ps][1])
            Bits_DictRound[f"{ps}"].append(statesRound[ps][2])
          statesRound = next_statesRound
          
          
          
          next_statesGameTheory, rewardGameTheory, doneGameTheory, terminalGameTheory = GameTheoryEnv.stepWithStepSize(actionsGameTheory, total_timesteps, g)
          next_statesGameTheory = {
                  i: np.array(list(next_statesGameTheory[0][f"ps{i}"]) + [next_statesGameTheory[1][f"ps{i}"]] + [next_statesGameTheory[2][f"ps{i}"]])
                  for i in range(1, nOfPs + 1)        
          }
          for ps in range(1, nOfPs + 1):
            if next_statesGameTheory[ps][2] != 0 :
              Max_StepsGameTheory[ps] += 1  
          for ps in range(1,nOfPs+1):
            Power_dictGameTheory[f"{ps}"].append(actionsGameTheory[ps])
            AoI_dictGameTheory[f"{ps}"].append(statesGameTheory[ps][1])
            Bits_DictGameTheory[f"{ps}"].append(statesGameTheory[ps][2])
          statesGameTheory = next_statesGameTheory



          total_timesteps += g
        average_power = 0 
        average_powerRound = 0
        average_powerGameTheory = 0
        print(Power_dictGameTheory)
        for ps in range(1, nOfPs + 1):
        #   print(sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}']))
          average_power += sum(Power_dict[f'{ps}'])/len(Power_dict[f'{ps}'])
          average_powerRound += sum(Power_dictRound[f'{ps}'])/len(Power_dictRound[f'{ps}'])
          average_powerGameTheory += sum(Power_dictGameTheory[f'{ps}'])/len(Power_dictGameTheory[f'{ps}'])
        average_powerRound = average_powerRound/nOfPs
        print(average_powerGameTheory)
        average_power = average_power/nOfPs
        average_powerGameTheory = average_powerGameTheory/nOfPs
        print(f"Average power usage of all PSs is {average_power}")
        print(f"Average Round Robbin power usage of all PSs is {average_powerRound}")
        print(f"Average GameTheory  power usage of all PSs is {average_powerGameTheory}")
        PowerList.append(average_power)
        PowerListRound.append(average_powerRound)
        PowerListGameTheory.append(average_powerGameTheory)
        AoI_Violoation_Probability = {}
        AoI_Violoation_ProbabilityRound = {}
        AoI_Violoation_ProbabilityGameTheory = {}
        for ps in range(1, nOfPs + 1):
          error = 0 
          errorRound = 0
          errorGameTheory = 0
          for index in range(len(AoI_dict[f"{ps}"])):
            if index <= len(AoI_dict[f"{ps}"]) -2 :
                if AoI_dict[f"{ps}"][index] > 2*int(w):
                  error += 1
          for index in range(len(AoI_dictRound[f"{ps}"])):
            if index <= len(AoI_dictRound[f"{ps}"]) -2 :
                if AoI_dictRound[f"{ps}"][index] > 2*int(w):
                  errorRound += 1
          for index in range(len(AoI_dictGameTheory[f"{ps}"])):
            if index <= len(AoI_dictGameTheory[f"{ps}"]) -2 :
                if AoI_dictGameTheory[f"{ps}"][index] > 2*int(w):
                  errorGameTheory += 1
          AoI_Violoation_Probability[ps] = (error/len( AoI_dict[f"{ps}"]))*100
          AoI_Violoation_ProbabilityRound[ps] = (errorRound/len( AoI_dictRound[f"{ps}"]))*100
          AoI_Violoation_ProbabilityGameTheory[ps] = (errorGameTheory/len( AoI_dictGameTheory[f"{ps}"]))*100
        mean = 0
        meanRound = 0
        meanGameTheory = 0
        for ps in range(1, nOfPs + 1):
            mean += AoI_Violoation_Probability[ps]
        for ps in range(1, nOfPs + 1):
            meanRound += AoI_Violoation_ProbabilityRound[ps]
        for ps in range(1, nOfPs + 1):
            meanGameTheory += AoI_Violoation_ProbabilityGameTheory[ps]

        mean_value = np.mean(list(AoI_Violoation_Probability.values()))
        std_dev = np.std(list(AoI_Violoation_Probability.values()))
        cov = std_dev / mean_value
        maxAoI = max(AoI_Violoation_Probability.values())
        print(f"Average AoI violation probability of all PSs is {mean_value}")
        print(f"CoV AoI violation probability of all PSs is {cov}")
        print(f"Standard Deviation of AoI violation probability of all PSs is {std_dev}")
        print(f"Max AoI violation probability of all PSs is {maxAoI}")

        print(f"Average Round Robbin AoI violation probability of all PSs is {meanRound/nOfPs}")
        print(f"Average Game Theory AoI violation probability of all PSs is {meanGameTheory/nOfPs}")

        AoIViolationLit.append(mean/nOfPs)
        AoIViolationLitRound.append(meanRound/nOfPs)
        AoIViolationLitGameTheory.append(meanGameTheory/nOfPs)
  print(AoIViolationLit)
  print(PowerList)
  print(AoIViolationLitRound)
  print(PowerListRound)
  print(AoIViolationLitGameTheory)
  print(PowerListGameTheory)
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
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))






plt.figure(11)

plt.title("Average AoI Violation for 15 Mbits and αᵢ = 0.1 and window = 60ms")
plt.plot([4,5,6,7,8,9], AoIViolationLit, label = "Proposed Algorithm",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], AoIViolationLitGameTheory , label = "Non Cooperative Game",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], AoIViolationLitRound , label = "Round Robin",  linestyle = "solid", marker = "^", color = "green")
plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))



plt.figure(12)

plt.title("Average Power Usage for 15 Mbits and αᵢ = 0.1 and window = 60ms")
plt.plot([4,5,6,7,8,9], PowerList, label = "Proposed Algorithm",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], PowerListGameTheory , label = "Non Cooperative Game",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], PowerListRound , label = "Round Robin",  linestyle = "solid", marker = "^", color = "green")
plt.xlabel("Number of PS")
plt.ylabel("Average Power Usage")
plt.axhline(y = 0.25, color = "red", linestyle = 'solid', label = "Maaximum Acceptable Average Power", )
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))






plt.figure(1)

plt.title("Average AoI Violation for 20 Mbits and αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [29, 40, 93, 70, 99, 99] , label = "window = 30ms, deadline = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [3.46,9.2,14,34.77,99,99] , label = "window = 40ms, deadline = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.16,1.63,6.76,17.29,99.35,99.39] , label = "window = 45ms, deadline = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.12,0.81,4.48,28.37,95,99] , label = "window = 50ms, deadline = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.25, 0.45, 2.42,1.28,98,99] , label = "window = 60ms, deadline = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(2)
plt.title("Average Power Usage for 20 Mbits and αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.08, 0.18, 0.94, 0.9, 0.99, 1] , label = "window = 30ms, deadline = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [0.09,0.16,0.12,0.09,0.96,0.99] , label = "window = 40ms, deadline = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.16,0.25,0.21,0.23,0.93,0.97] , label = "window = 45ms, deadline = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.09,0.2,0.13,0.13,0.88,1] , label = "window = 50ms, deadline = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.21, 0.25,0.12,0.14,0.98,0.99] , label = "window = 60ms, deadline = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.axhline(y = 0.25, color = "red", linestyle = 'solid', label = "Maximum Acceptable Average Power", )
plt.legend(loc = "best")
plt.xlabel("Number of PS")
plt.ylabel("Average Power Usage")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(3)

plt.title("Average AoI Violation for 15 Mbits and αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [30, 33, 34, 51, 60, 77] , label = "window = 30ms, deadline = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [7.91,13,26,29,34,61] , label = "window = 40ms, deadline = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [1.88,10,5.8,25.9,34,99] , label = "window = 45ms, deadline = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.35,0.43,0.18,14,26,99] , label = "window = 50ms, deadline = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.29, 0.18, 3.2,0.11,99,99] , label = "window = 60ms, deadline = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )
plt.legend(loc = "best")
plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.figure(4)
plt.title("Average Power Usage for 15 Mbits and  αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.09, 0.15, 0.06, 0.15, 0.044, 0.1] , label = "window = 30ms, deadline = 60ms",  linestyle = "solid", marker = "o", color = "blue")
plt.plot([4,5,6,7,8,9], [0.23,0.28,0.14,0.10,0.17,0.16] , label = "window = 40ms, deadline = 80ms",  linestyle = "solid", marker = "*", color = "orange")
plt.plot([4,5,6,7,8,9], [0.19,0.17,0.21,0.2,0.32,0.99] , label = "window = 45ms, deadline = 90ms",  linestyle = "solid", marker = "^", color = "green")
plt.plot([4,5,6,7,8,9], [0.08,0.14,0.23,0.1,0.2,0.99] , label = "window = 50ms, deadline = 100ms",  linestyle = "solid", marker = "s", color = "aqua")
plt.plot([4,5,6,7,8,9], [0.11, 0.3,0.21,0.20,0.99,0.96] , label = "window = 60ms, deadline = 120ms",  linestyle = "solid", marker = "<", color = "m")
plt.axhline(y = 0.25, color = "red", linestyle = 'solid', label = "Maximum Acceptable Average Power", )

plt.legend(loc = "best")
plt.xlabel("Number of PS")
plt.ylabel("Average Power Usage")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(5)
plt.title("Average AoI Violation Comparison, window = 30ms, deadline = 60ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [30, 33, 34, 51, 60, 77] , label = "15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [29, 40, 93, 70, 99, 99] , label = "20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(6)
plt.title("Average AoI Violation Comparison, window = 40ms, deadline = 80ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [7.91,13,20,29,34,61] , label = "15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [3.46,9.2,14,34.77,99,99] , label = "20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable aoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(7)
plt.title("Average AoI Violation Comparison, window = 45ms, deadline = 90ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [1.88,10,5.8,25.9,34,99] , label = "15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.16,1.63,6.76,17.29,99.35,99.39] , label = "20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(8)
plt.title("Average AoI Violation Comparison, window = 50ms, deadline = 100ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.35,0.43,0.18,14,26,99] , label = "15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.12,0.81,4.48,28.37,95,99] , label = "20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
plt.legend(loc = "best")
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

plt.figure(9)
plt.title("Average AoI Violation Comparison, window = 60ms, deadline = 120ms, αᵢ = 0.1")
plt.plot([4,5,6,7,8,9], [0.29, 0.18, 3.2,0.11,99,99] , label = "15 Mbits",  linestyle = "solid", marker = "o", color = "m")
plt.plot([4,5,6,7,8,9], [0.25, 0.45, 2.42,1.28,98,99], label = "20 Mbits",  linestyle = "solid", marker = "s", color = "green")
plt.axhline(y = 15, color = "red", linestyle = 'solid', label = "Maximum Acceptable AoI Violation Pr", )

plt.xlabel("Number of PS")
plt.ylabel("Probability of AoI Violation")
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
