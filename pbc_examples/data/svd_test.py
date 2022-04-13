import matplotlib.pyplot as plt
from numpy.linalg import linalg
import numpy as np

from pbc_examples.data.plot import plot_solution_1d
from pbc_examples.data.simple_pde import SimplePDEDataset



dataset = SimplePDEDataset([
    # ('convection', 0, 1, 0, 'np.sin(10*x)', 200, 200, 0),
    ('convection', 0, 5, 0, 'np.exp(-np.power((x - 0.5*np.pi)/(np.pi/16), 2.)/2.)', 200, 199, 0),
])

d = dataset[0]
x = d['u'].squeeze(-1)
# print(x.shape)
u, s, vh = linalg.svd(x, full_matrices=False)

# print(u.shape, s.shape, vh.shape)
# sd = np.hstack([np.diag(s), np.zeros()
# print(sd.shape)
# print(np.abs(u.dot(np.diag(s)).dot(vh) - x).max())
# print((u[0].reshape(-1, 1) * s.reshape((-1, 1)) * vh).shape)

# exit()
# print(s[..., 0])
plot_solution_1d(x, d['x'], d['t'])
plt.imshow(x)
plt.show()

plt.title('time')
for i in range(len(s)):
    plt.plot(np.real(u[..., i] * s[i]))
plt.show()

plt.title('space')
for i in range(len(s)):
    plt.plot(np.real(vh[i] * s[i]))
plt.show()

vhs = u[0].reshape(-1, 1) * s.reshape((-1, 1)) * vh
plt.title('space t0')
for i in range(len(s)):
    plt.plot(np.real(vhs[i]))
plt.plot(vhs.sum(0))
plt.show()


plt.title('modes')
plt.plot(np.abs(s))
plt.show()



# uu, su, vhu = linalg.svd(u.dot(np.diag(s)))
# # print(uu.shape, su.shape, vhu.shape, uu.max(), np.abs(uu).mean())
# plt.title('timesqss')
# for i in range(len(su)):
#     plt.plot(np.real(uu[..., i] * su[i]))
# plt.show()
#
# plt.title('dims')
# for i in range(len(su)):
#     plt.plot(np.real(vhu[i] * su[i]))
# plt.show()
#
# print(su.max(), su.min(), su.mean())
# plt.show()