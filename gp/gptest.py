import numpy as np
import gp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


N = 100
train_x1 = np.arange(N).reshape(N, 1) / (1.0 * N) * 1
train_x2 = np.random.rand(N).reshape(N, 1)
train_x3 = np.random.rand(N).reshape(N, 1)
train_x = np.hstack((train_x1, train_x2))
train_x = np.hstack((train_x, train_x3))

train_y = np.sin(train_x1.ravel() * 2 * np.pi) + (2*np.random.rand(N)-1)*.1 + np.cos(train_x2.ravel() * 2 * np.pi) * .3

test_x1 = np.arange(1000).reshape(1000, 1) / 1000.
test_x2 = np.random.rand(1000).reshape(1000, 1)
test_x3 = np.random.rand(1000).reshape(1000, 1)
test_x = np.hstack((test_x1, test_x2))
test_x = np.hstack((test_x, test_x3))

test_y = np.sin(test_x1.ravel() * 2 * np.pi) + np.cos(test_x2.ravel() * 2 * np.pi) * .3
clf = gp.GP(alpha=1, beta=1e-2, kernel_name="Matern52", params=10)

clf.fit(train_x, train_y, hyper_opt_iter=50)

mean, var = clf.decision_function(test_x)

print np.sum((mean - test_y)**2)
print clf.theta
sig = np.sqrt(var)
print sig.shape
print mean.shape

plt.clf()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test_x[:, 0], test_x[:, 1], mean, c='blue')
ax.scatter(test_x[:, 0], test_x[:, 1], test_y, c='red')
plt.grid()
plt.savefig("gp.pdf")
