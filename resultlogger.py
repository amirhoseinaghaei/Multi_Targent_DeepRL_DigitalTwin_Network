import json 
import matplotlib.pyplot as plt 
from Environment import SimulationParams


f = open("./pytorch_models/5ps_3ch_centralcritic_100bits_35w_70d/Results.json")
data0 = json.load(f)
f = open("./pytorch_models/5ps_8ch_centralcritic_100bits_35w_70d/Results.json")
data1 = json.load(f)
f = open("./pytorch_models/5ps_15ch_centralcritic_100bits_35w_70d_2/Results.json")
data2 = json.load(f)
f = open("./pytorch_models/5ps_20ch_centralcritic_100bits_35w_70d_2/Results.json")
data3 = json.load(f)
f = open("./pytorch_models/5ps_25ch_centralcritic_100bits_35w_70d/Results.json")
data4 = json.load(f)
NumberOfPS = SimulationParams.NumberOfPS



plt.figure(1)
plt.title("RL scheduler convergence time, Nₖ = 3")
for i in range(1,NumberOfPS+1):
    print(i)
    plt.plot(range(200, (len(data0[f"{i}"])+1)*200,200), data0[f"{i}"], label = f"PS{i}", linestyle = "-.")

plt.legend(loc ="best")
plt.figure(2)
plt.title("RL scheduler convergence time, Nₖ = 8")
for i in range(1,NumberOfPS+1):
    print(i)
    plt.plot(range(200, (len(data1[f"{i}"])+1)*200,200), data1[f"{i}"], label = f"PS{i}", linestyle = "-.")

plt.legend(loc ="best")
plt.figure(3)
plt.title("RL scheduler convergence time, Nₖ = 15")
for i in range(1,NumberOfPS+1):
    print(i)
    plt.plot(range(200, (len(data2[f"{i}"])+1)*200,200), data2[f"{i}"], label = f"PS{i}", linestyle = "-.")

plt.legend(loc ="best")
plt.figure(4)
plt.title("RL scheduler convergence time, Nₖ = 20")
for i in range(1,NumberOfPS+1):
    print(i)
    plt.plot(range(200, (len(data3[f"{i}"])+1)*200,200), data3[f"{i}"], label = f"PS{i}", linestyle = "-.")
# plt.axhline(y = 14000, color = "orange", linestyle = 'solid', label = "boundry", )

plt.legend(loc ="best")
plt.figure(5)
plt.title("RL scheduler convergence time, Nₖ = 25")
for i in range(1,NumberOfPS+1):
    print(i)
    plt.plot(range(200, (len(data4[f"{i}"])+1)*200,200), data4[f"{i}"], label = f"PS{i}", linestyle = "-.")
# plt.axhline(y = 14000, color = "orange", linestyle = 'solid', label = "boundry", )

plt.legend(loc ="best")
plt.show()
