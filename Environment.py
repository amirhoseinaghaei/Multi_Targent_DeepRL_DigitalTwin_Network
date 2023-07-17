import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel


SimulationParams = SimulationParameters("Configs.json")
SimulationParams.Configure()
NumberOfPS = SimulationParams.NumberOfPS
NumberOfTch = SimulationParams.NumberOfTCh
NumberOfBits = SimulationParams.NumberOfBits
NumberOfCPUCycles = SimulationParams.NumberOfCpuCycles
TotalCPUCapacity = SimulationParams.TotalCPUCapacity
windows = SimulationParams.windows
startupdate = SimulationParams.startupdate
deadlines = SimulationParams.deadlines
AoI_sensitivity = SimulationParams.AoI_sensitivity

class CustumEnv(gym.Env):
   
    def __init__(self, NumOfPS):
      self.Wireless_Tr_Channels = WirelessChannel(0.001,100 , NumOfPSs= NumberOfPS)
      self.AoIs_Dict = dict()
      self.Data_Dict = dict()
      self._ps_gain = dict()
      self._ps_AoI = dict()
      self._ps_U = dict()
      self.NumOfPS = NumOfPS 
      self.start_update = dict()
      self.windows = dict()
      self.deadlines = dict()
      for i in range(self.NumOfPS):
          self.windows[i+1] = windows[i]
          self.start_update[i+1] = startupdate[i]
          self.deadlines[i+1] = deadlines[i]
      self.observation_space = self.create_initial_observation()
      
      

      self.action_space = spaces.Dict(
         {
            "T-Power": spaces.Box(0,150, shape=(NumberOfTch,), dtype= np.float32),
            "C-Power": spaces.Box(0,1, shape=(1,), dtype= np.float32)

         }
      )
    def create_initial_observation(self):
        observation = spaces.Dict({})
        for n in range(self.NumOfPS):
            observation[f"PS{n}"] = spaces.Dict(
                    {   
                        "Channel_Gain" : spaces.Box(0,10, shape=(NumberOfTch,), dtype= np.float32),
                        "AoI" : spaces.Box(0,100, shape=(1,), dtype= np.float32),
                        "U" : spaces.Box(0,100, shape=(1,), dtype= np.int32)
                    }
                )
        return observation
    def _get_obs(self , ps):
        return {"Channel_Gain": self._ps_gain[f"ps{ps}"], "AoI": self._ps_AoI[f"ps{ps}"], "U": self._ps_U[f"ps{ps}"]}
    def _get_info(self, ps):
        new_dict = copy.deepcopy(self._ps_gain)
        # new_dict = new_dict.pop(f"ps{ps}" , None)
        del new_dict[f"ps{ps}"]
        return {"Interference_Channel_Gain": new_dict}
    def reset(self, ps, seed = None, options = None):
        if ps <= self.NumOfPS:
            self._ps_gain[f"ps{ps}"] = self.np_random.uniform(0, 1,3)
            self._ps_AoI[f"ps{ps}"] = 0
            self._ps_U[f"ps{ps}"] = 1100
            self.AoIs_Dict[ps] = [0]
            self.Data_Dict[ps] = [1100]

            observation = self._get_obs(ps)

            # return observation

        return None
    def channel_gains_transition(self, ps):
        self._ps_gain[f"ps{ps}"] = self.np_random.uniform(0,1,3)           

    def step(self, action, time):
        self.E = []
        reward = dict()
        for ps in range(1, self.NumOfPS +1 ):
            print(f"-------------time is: {time}--------------------")

            if self._ps_U[f"ps{ps}"] == 0 and time%(self.windows[ps]) != self.start_update[ps]:
                print(f"current state ps:{ps}: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']})") 
                if (time+1)%(self.windows[ps]) == self.start_update[ps]:
                    self._ps_U[f"ps{ps}"] = NumberOfBits + NumberOfCPUCycles
                self._ps_AoI[f"ps{ps}"] += 1
                self.channel_gains_transition(ps)
                self.AoIs_Dict[ps].append(self._ps_AoI[f"ps{ps}"])
                self.Data_Dict[ps].append(self._ps_U[f"ps{ps}"])
                print(f" PS: {ps} => Next state: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']}), Reward:{0}")
                print("\n\n")


            elif time%(self.windows[ps]) ==  self.start_update[ps] and time >= self.start_update[ps]:
                print(f"current state ps:{ps}: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']})") 
                self._ps_AoI[f"ps{ps}"] += 1
                self.AoIs_Dict[ps].append(self._ps_AoI[f"ps{ps}"])
                rest_action = copy.deepcopy(action)
                rest_action.pop(ps)
                self._ps_U[f"ps{ps}"] = np.maximum(NumberOfCPUCycles,self._ps_U[f"ps{ps}"] - self.Wireless_Tr_Channels.calculate_transmission_rate(channel_gain= list(self._get_obs(ps)["Channel_Gain"]), interference_gain = self._get_info(ps)["Interference_Channel_Gain"] , interference_power = rest_action, power= action[ps]))
                self.channel_gains_transition(ps)
                self.Data_Dict[ps].append(self._ps_U[f"ps{ps}"])
                print(f" PS{ps} Next state: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']}), Reward : {self.stochastic_reward(action=action[ps], AoI=self._ps_AoI[f'ps{ps}'], AoI_weight= 1, PS = ps )}")
                print("\n\n")


            elif NumberOfCPUCycles < self._ps_U[f"ps{ps}"] <= NumberOfBits + NumberOfCPUCycles and time%(self.windows[ps]) != self.start_update[ps] and time >= self.start_update[ps]:
                print(f"current state ps:{ps}: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']})") 
                self._ps_AoI[f"ps{ps}"] += 1
                self.AoIs_Dict[ps].append(self._ps_AoI[f"ps{ps}"])
                rest_action = copy.deepcopy(action)
                rest_action.pop(ps)
                self._ps_U[f"ps{ps}"] = np.maximum(NumberOfCPUCycles,self._ps_U[f"ps{ps}"] - self.Wireless_Tr_Channels.calculate_transmission_rate(channel_gain= list(self._get_obs(ps)["Channel_Gain"]), interference_gain = self._get_info(ps)["Interference_Channel_Gain"] , interference_power = rest_action, power= action[ps]))
                if (time+1)%(self.windows[ps]) == self.start_update[ps]:
                    self._ps_U[f"ps{ps}"] = NumberOfBits + NumberOfCPUCycles
                self.channel_gains_transition(ps)
                self.Data_Dict[ps].append(self._ps_U[f"ps{ps}"])
                print(f" PS{ps} Next state: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']}), Reward : {self.stochastic_reward(action=action[ps], AoI=self._ps_AoI[f'ps{ps}'], AoI_weight= 1, PS = ps )}")
                print("\n\n")



                # print(f"Remaining bits of {ps}th ps is: {self._ps_U[f'ps{ps}']}")
            elif 0 < self._ps_U[f"ps{ps}"] <= NumberOfCPUCycles and time%(self.windows[ps]) != self.start_update[ps]:
                print(f"current state ps:{ps}: (AoI: {self._ps_AoI[f'ps{ps}']}, U: {self._ps_U[f'ps{ps}']}, H:{self._ps_gain[f'ps{ps}']})") 
                self.channel_gains_transition(ps)
                self._ps_AoI[f"ps{ps}"] += 1
                self.E.append(ps)
        self.P = 1
        while self.P != 0:
            for n in self.E: 
                # Determinisitic CPU resource allocation
                f_n = self.deterministicCPUResourceAllocation(n, time)
                # print(f"Allocated resource {f_n}")
                self._ps_U[f"ps{n}"] = max(0,self._ps_U[f"ps{n}"] - TotalCPUCapacity*f_n)
                self.Data_Dict[n].append(self._ps_U[f"ps{n}"])

                if self._ps_U[f"ps{n}"] == 0 and self._ps_AoI[f'ps{n}'] > self.windows[n]:
                     self._ps_AoI[f'ps{n}'] = time  - (((time- self.start_update[n]) // self.windows[n])*self.windows[n] + self.start_update[n]) + 1
                self.AoIs_Dict[n].append(self._ps_AoI[f"ps{n}"])
                
                print(f"PS: {n} => Next state: (AoI: {self._ps_AoI[f'ps{n}']}, U: {self._ps_U[f'ps{n}']}, H:{self._ps_gain[f'ps{n}']}), Reward: {self.deterministic_reward(AoI= self._ps_AoI[f'ps{n}'], PS= n)}")
                print("\n\n")

                # print(f"PS-Dt{n}")
            if len(self.E) == 0:
                break    
            
    def stochastic_reward(self, action, AoI, AoI_weight , PS):
        reward = 0
        for i in (action):
            reward += i
        reward = -1*reward
        reward = reward - AoI_weight*np.maximum(0,np.exp(-AoI_sensitivity*(self.deadlines[PS] - AoI))-1)
        return reward
    def deterministic_reward(self, AoI, PS):
        reward =  -np.maximum(0,np.exp(-AoI_sensitivity*(self.deadlines[PS] - AoI))-1)
        if reward == -0:
            reward = 0
        return reward
    def reset_all_agents():
      TotalCPU  = TotalCPUCapacity
    def deterministicCPUResourceAllocation(self, ps, time):
        most_significant_ps_dt = 1
        val = self.deadlines[1] - self._ps_AoI[f"ps{1}"]
        for n in self.E:
            if  self.deadlines[n] - self._ps_AoI[f"ps{n}"] < val:
                most_significant_ps_dt = n
                val = self.deadlines[n] - self._ps_AoI[f"ps{n}"]
        if ps == n:
            f_n = min((self._ps_U[f"ps{ps}"]/(TotalCPUCapacity*self.P)),self.P)

            self.P = max(0,self.P-f_n)
            self.E.remove(ps)
        else:
            f_n = 0
        return f_n