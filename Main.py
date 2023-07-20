import copy
import gym 
from gym import spaces
import numpy as np
from Config import SimulationParameters
from WirelessChannel import WirelessChannel
from Environment import CustumEnv , SimulationParams
NumberOfPS = SimulationParams.NumberOfPS
Env = CustumEnv(NumberOfPS)
for i in range(NumberOfPS):
   Env.reset(i+1)
  #   

for t in range(1,50):
  action = {1:[np.random.uniform(20,30),np.random.uniform(20,30),np.random.uniform(20,50)] ,
           2:[np.random.uniform(10,100),np.random.uniform(10,100),np.random.uniform(10,50)],
           3:[np.random.uniform(10,100),np.random.uniform(10,100),np.random.uniform(10,50)],
               4:[np.random.uniform(10,110),np.random.uniform(10,110),np.random.uniform(10,50)],
                 5:[np.random.uniform(10,100),np.random.uniform(10,100),np.random.uniform(10,50)]}
  print(action)
  Env.step(action,t)
from matplotlib import pyplot as plt



print(Env.AoIs_Dict[1])
AoIs = dict()
times = dict()
Data = dict()
for i in range(1,NumberOfPS+1):
  AoIs[i] = []
  times[i] = []
  Data[i] = []
for i in range(1,NumberOfPS+1):
   for j in Env.AoIs_Dict[i]:
      AoIs[i].append(j[0])
      times[i].append(j[1])
   for j in Env.Data_Dict[i]:
      Data[i].append(j[0])
      # times[i].append(j[1])

plt.figure(num=1)
for i in range(1,NumberOfPS+1):
    plt.plot(times[i], AoIs[i], linestyle = 'dashed', label = f"ps {i}")
plt.axhline(y = 20, color = "red", linestyle = 'solid', label = "deadline", )
plt.legend()
plt.figure(num=2)
for i in range(1,NumberOfPS+1):
    plt.plot(times[i], Data[i], label = f"ps {i}")
plt.legend()
plt.show()