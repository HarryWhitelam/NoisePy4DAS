import matplotlib.pyplot as plt
import numpy as np

corr = np.random.randint(25, size=(1000, 100))
cha1, cha2 = 1000, 1099
step = 4

corr = corr[:, ::step]
print(corr.shape)

plt.figure(figsize = (12, 5), dpi = 150)
plt.imshow(corr[:, :].T, aspect = 'auto', cmap = 'bwr', 
           vmax = 25, vmin = 0, origin = 'lower', interpolation=None)

print((np.linspace(cha1, cha2, 4) - cha1)/4)
print(np.linspace(cha1, cha2, 4))
_ = plt.yticks((np.linspace(cha1, cha2, 4) - cha1)/4, 
               [int(i) for i in np.linspace(cha1, cha2, 4)], fontsize = 12)
plt.ylabel("Channel number", fontsize = 16)
_ = plt.xticks(np.arange(0, 1601, 200), (np.arange(0, 801, 100) - 400)/50, fontsize = 12)
plt.xlabel("Time lag (sec)", fontsize = 16)

plt.show()
