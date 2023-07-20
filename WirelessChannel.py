import math
import time
import numpy as np 


class WirelessChannel():
    def __init__(self, N0, B , NumOfPSs):
        self.N0 = N0
        self.B = B
        self.NumOfPSs  = NumOfPSs
        self.rou = 0

    def generate_new_channel_gain(self, NumberOfTch, ps_gain):
        print(self.rou * ps_gain + np.random.uniform(0,1-math.pow(self.rou,2),size=3))
        return self.rou * ps_gain + np.random.uniform(0,1-math.pow(self.rou,2),size=3)
    def calculate_transmission_rate(self, channel_gain, interference_gain, interference_power, power):
        rate = 0
        for i in range(len(channel_gain)):
            rate += np.log(1 + power[i]*channel_gain[i] /(self.calculate_interference(interference_gain, interference_power, i) + self.N0*self.B))
        return self.B*rate
    def calculate_interference(self, interference_gain, interference_power, i):
        interference = 0
        for key in interference_power.keys():
            interference += interference_gain[f"ps{key}"][i]* interference_power[key][i]
        return interference
    

# wc = WirelessChannel(0.5,10)
# print(wc.calculate_transmission_rate([1,3], [0.01,0.08], [10,15], [100,150]))