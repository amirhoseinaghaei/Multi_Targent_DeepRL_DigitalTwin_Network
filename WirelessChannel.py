import json
import math
from re import T
import time
import numpy as np 


class WirelessChannel():
    def __init__(self, N0, B , NumOfPSs, NumOfGainstates, psType, GainMapping, GainProbMapping):
        self.N0 = N0
        self.B = B
        self.NumOfPSs  = NumOfPSs
        self.rou = 0
        self.gain_transition_matrix = {}
        self.gain_list = {}
        self.NumOfGainstates = NumOfGainstates
        self.psType = psType
        self.GainMapping = GainMapping
        self.GainProbMapping = GainProbMapping
    def generate_gain_list(self):
        for ps in range(1,self.NumOfPSs+1):
            self.gain_list[ps] = []
            for i in range(self.NumOfGainstates):   
                self.gain_list[ps].append(self.GainMapping[self.psType[ps-1]][i])
                # self.gain_list[ps].append((ps*3)+ 4*(i))

    def generate_transition_probability_matrix(self):
        for ps in range(1,self.NumOfPSs+1):
            self.gain_transition_matrix[ps] = []
            for i in range(self.NumOfGainstates):
                row = []     
                for y in range(self.NumOfGainstates):
                    row.append(self.GainProbMapping[self.psType[ps-1]][y])
                self.gain_transition_matrix[ps].append(row)
        return self.gain_transition_matrix

    # def generate_transition_probability_matrix(self):
    #     for ps in range(1,self.NumOfPSs+1):
    #         self.gain_transition_matrix[ps] = []
    #         for i in range(self.NumOfGainstates):
    #             row = []
    #             for y in range(self.NumOfGainstates):
    #                 # ((y+1)*0.3)/self.gen_Sum()
    #                 row.append(((y+0.5)*0.2)/self.gen_Sum())
    #             self.gain_transition_matrix[ps].append(row)
    # #    print(len(self.gain_transition_matrix[1]))
    #     return self.gain_transition_matrix
    # def gen_Sum(self):
    #     sum = 0
    #     for i in range(self.NumOfGainstates):
    #         sum += (i+0.5)*(0.2)
    #     return sum

    def generate_new_channel_gain(self,NumberOfTch, ps_gain, ps):
        random_number = np.random.uniform(0,1,1)    
        index = self.gain_list[ps].index(ps_gain[0])
        prob_sum = 0
        i = 0
        while True:
            prob_sum += self.gain_transition_matrix[ps][index][i]
            if random_number <= prob_sum:
                return [self.gain_list[ps][i]]
            i += 1
    def calculate_transmission_rate(self, channel_gain, interference_gain, interference_power, power):
        rate = 0
        for i in range(len(channel_gain)):
            channelGain = 10**(channel_gain[i]/10)
            rate += math.log10(1 + (power[i]*channelGain /(self.calculate_interference(interference_gain, interference_power, i) + self.N0*self.B)))
        if math.isnan(rate):
        # Handle the NaN case appropriately, maybe set a default value or raise an error
            rate = 0
        return int((self.B*rate)/1e6)
    def calculate_transmission_rate2(self, channel_gain, interference_gain, interference_power, power):
        rate = 0
        for i in range(len(channel_gain)):
            channelGain = 10**(channel_gain[i]/10)
            rate += math.log10(1 + (power[i]*channelGain /(self.calculate_interference(interference_gain, interference_power, i) + self.N0*self.B)))
        return rate
    def calculate_interference(self, interference_gain, interference_power, i):
        # print(interference_power)
        # print(interference_gain)
        # print(i)
        # time.sleep(10)

        interference = 0
        for key in interference_power.keys():
            interferenceG = 10**(interference_gain[f"ps{key}"][i]/10)
            interference += interferenceG * interference_power[key][i]
        return interference
    
# from Config import SimulationParameters
# SimulationParams = SimulationParameters("Configs.json")
# SimulationParams.Configure()
# wc = WirelessChannel(1e-17,5e9, SimulationParams.NumberOfPS, SimulationParams.NumOfGainStates, SimulationParams.psType,SimulationParams.GainMapping, SimulationParams.GainProbabilityMapping)
# wc.generate_gain_list()
# wc.generate_transition_probability_matrix()
# print(wc.calculate_transmission_rate([1,3], [0.01,0.08], [10,15], [100,150]))