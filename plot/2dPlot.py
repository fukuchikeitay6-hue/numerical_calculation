import numpy as np
import matplotlib.pyplot as plt

loader = np.load("./data/D_1e-6_0.1-0.15.npz", allow_pickle=True)
data = loader["data"]
print(data.shape)
params = loader["params"].item()
spf = params["spf"]
numFrame = len(data)
t = np.linspace(0, spf*(numFrame-1), numFrame)

pos1 = 200
# pos2 = 300
pos3 = 499

data1 = data[:, pos1, 0].reshape(numFrame)
# data2 = data[:, pos2, 0].reshape(numFrame)
data3 = data[:, pos3, 0].reshape(numFrame)


fig, ax = plt.subplots()
ax.set_xlim(0.1, 0.8)
# ax.set_yscale('log')
ax.plot(t, data1, label=f"pos: {pos1}", c="b", linewidth=3)
# ax.plot(t, data2, label=f"pos: {pos2}")
ax.plot(t, data3, label=f"pos: {pos3}", c="r", linewidth=3)

# plt.legend()
# plt.show()
plt.close()
fig.savefig("/Users/fukuchikeita/Desktop/学振/figure/fig1.png", transparent=True)