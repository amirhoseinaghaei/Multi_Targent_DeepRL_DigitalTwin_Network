import json 
import matplotlib.pyplot as plt 
from Environment import SimulationParams
import pandas as pd

f = open("./pytorch_models/5ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward/Results.json")
data0 = json.load(f)
f = open("./pytorch_models/5ps_100ch_centralcritic_100bits_40w_80d_power_reward_powermonitor_paper/Results2.json")
data1 = json.load(f)
f = open("./pytorch_models/5ps_100ch_centralcritic_100bits_40w_80d_power_reward_powermonitor_paper/Results3.json")
data2 = json.load(f)
f = open("./results/Results3.json")
data3 = json.load(f)
f = open("./results/Results2.json")
data4 = json.load(f)
f = open("./pytorch_models/6ps_3ch_centralcritic_100bits_40w_80d_power_simple_reward_powermonitor_paper/Results.json")
data5 = json.load(f)
NumberOfPS = SimulationParams.NumberOfPS



# plt.figure(1)
# plt.title("RL scheduler convergence time, Nₖ = 3")
# for i in range(1,NumberOfPS+1):
#     print(i)
#     plt.plot(range(200, (len(data0[f"{i}"])+1)*200,200), data0[f"{i}"], label = f"PS{i}", linestyle = "-.")
# plt.axhline(y = 15000, color = "orange", linestyle = 'solid', label = "boundry", )

# plt.legend(loc ="best")
# plt.figure(2)
# plt.title("RL scheduler convergence time, Nₖ = 8")
# for i in range(1,NumberOfPS+1):
#     print(i)
#     s = pd.Series(data1[f"{i}"])
#     data1[f"{i}"] = s.apply(lambda x: x/200)
#     plt.plot(range(200, (len(data1[f"{i}"])+1)*200,200), data1[f"{i}"], label = f"PS{i}", linestyle = "-.")
#     plt.xlabel("Episode")
#     plt.ylabel('Average AoI')
# plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "boundry", )

# plt.legend(loc ="best")
# plt.figure(3)
# plt.title("RL scheduler convergence time, Nₖ = 15")
# for i in range(1,NumberOfPS+1):
#     print(i)
#     # s = pd.Series(data2[f"{i}"])
#     # data2[f"{i}"] = s.apply(lambda x: x/200)
#     plt.plot(range(200, (len(data2[f"{i}"])+1)*200,200), data2[f"{i}"], label = f"PS{i}", linestyle = "-.")
#     plt.xlabel("Episode")
#     plt.ylabel('Average AoI')
# plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "boundry", )

# plt.legend(loc ="best")
plt.figure(4)
plt.title("RL scheduler AoI convergence time, Nₖ = 50")
for i in range(1,NumberOfPS+1):
    print(i)
    s = pd.Series(data3[f"{i}"])
    data3[f"{i}"] = s.apply(lambda x: x/200)
    plt.plot(range(200, (len(data3[f"{i}"])+1)*200,200), data3[f"{i}"], label = f"PS{i}", linestyle = "-.")
    plt.xlabel("Episode")
    plt.ylabel('Average AoI')
# plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "boundry", )

plt.legend(loc ="best")
plt.figure(5)
plt.title("RL scheduler power convergence time, Nₖ = 50")
for i in range(1,NumberOfPS+1):
    print(i)
    s = pd.Series(data4[f"{i}"])
    data4[f"{i}"] = s.apply(lambda x: x/200)
    plt.plot(range(200, (len(data4[f"{i}"])+1)*200,200), data4[f"{i}"] ,label = f"PS{i}", linestyle = "-.")
    plt.xlabel("Episode")
    plt.ylabel('Average power')
# plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "boundry", )

plt.legend(loc ="best")



# sum = data5[f"{1}"]
# for i in range(2,NumberOfPS+1):
#     sum = [a + b for a, b in zip(sum, data5[f"{i}"])]
# sum = [a/5 for a in sum]

# sum1 = data4[f"{1}"]
# for i in range(2,NumberOfPS+1):
#     sum1 = [a + b for a, b in zip(sum1, data4[f"{i}"])]
# sum1 = [a/5 for a in sum1]

# plt.figure(6)
# plt.title("RL scheduler convergence time, Nₖ = 35")

# print(i)
    # s = pd.Series(data5[f"{i}"])
    # data5[f"{i}"] = s.apply(lambda x: x/200)
# plt.plot(sum, label = f"reward shaping", linestyle = "-.")
# plt.plot(sum1, label = f"old", linestyle = "-.")

# plt.xlabel("Episode")
# plt.ylabel('Average AoI')
# plt.axhline(y = 80, color = "orange", linestyle = 'solid', label = "boundry", )

# plt.legend(loc ="best")


plt.show()
