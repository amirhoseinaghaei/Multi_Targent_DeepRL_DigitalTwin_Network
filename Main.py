import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
NumberOfPS = SimulationParams.NumberOfPS
Env = CustumEnv(NumberOfPS)
Env.reset(1)
Env.reset(2)
action = {1:[20,150,10] , 2:[10000,2000,30000]}
for t in range(1,50):
  Env.step(action,t)
from matplotlib import pyplot as plt

plt.figure(num=1)
for i in range(1,NumberOfPS+1):
    plt.plot(Env.AoIs_Dict[i], label = f"ps {i}")
plt.axhline(y = 20, color = "red", linestyle = '-', label = "deadline")
plt.legend()
plt.figure(num=2)
for i in range(1,NumberOfPS+1):
    plt.plot(Env.Data_Dict[i], label = f"ps {i}")
plt.legend()
plt.show()