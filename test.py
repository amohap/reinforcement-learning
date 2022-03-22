from matplotlib import pyplot as plt
import numpy as np
import os

np.random.seed(99)

X = np.random.rand(100)
Y = 2*X + 0.5*np.random.rand(100)

x=2

filename = os.path.splitext(__file__)[0]

x=0.1
y=0.5
z=0.7


path="experiments/{}/x{}_y{}_z{}/haha.png".format(filename, x, y, z)
# print(path)
# print(filename)

plt.scatter(X,Y)
# plt.show()
plt.savefig(path)