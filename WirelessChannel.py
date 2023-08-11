import math
import time
import numpy as np 


class WirelessChannel():
    def __init__(self, N0, B , NumOfPSs):
        self.N0 = N0
        self.B = B
        self.NumOfPSs  = NumOfPSs
        self.rou = 0
        self.gain_transition_matrix = {}
        self.gain_list = {}
        self.NumOfGainstates = 3
        self.variance = [0.01,0.02,0.03,0.02,0.01]
    def generate_gain_list(self):

        # for ps in range(1,self.NumOfPSs+1):
        #     self.gain_list[ps] = []
        #     for i in range(self.NumOfGainstates):
        #         self.gain_list[ps].append(np.sqrt(-2*((self.variance[ps-1]))*np.log(1-(i/self.NumOfGainstates))))
        # return Channel_gains
        for ps in range(1,self.NumOfPSs+1):
            self.gain_list[ps] = []
            for i in range(self.NumOfGainstates):
                self.gain_list[ps].append((ps*2)+ 4*(i))
        print(self.gain_list)
    def generate_transition_probability_matrix(self):
        for ps in range(1,self.NumOfPSs+1):
            self.gain_transition_matrix[ps] = []
            for i in range(self.NumOfGainstates):
                row = []
                for y in range(self.NumOfGainstates):
                    # ((y+1)*0.3)/self.gen_Sum()
                    row.append(((y+0.5)*0.2)/self.gen_Sum())
                self.gain_transition_matrix[ps].append(row)
        print(self.gain_transition_matrix)
    def gen_Sum(self):
        sum = 0
        for i in range(self.NumOfGainstates):
            sum += (i+0.5)*(0.2)
        return sum
    def generate_interference_channel_gain(self):
        return np.random.uniform(1,2,size=1)
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
            rate += math.log10(1 + (power[i]*channel_gain[i] /(self.calculate_interference(interference_gain, interference_power, i) + self.N0*self.B)))
        return self.B*rate
    def calculate_interference(self, interference_gain, interference_power, i):
        interference = 0
        for key in interference_power.keys():
            
            interference += interference_gain[f"ps{key}"][i] * interference_power[key][i]
        return interference
    

# wc = WirelessChannel(0.5,10)
# print(wc.calculate_transmission_rate([1,3], [0.01,0.08], [10,15], [100,150]))