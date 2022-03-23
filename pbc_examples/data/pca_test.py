import matplotlib.pyplot as plt
from numpy.linalg import linalg
import numpy as np

from pbc_examples.data.simple_pde import SimplePDEDataset



dataset = SimplePDEDataset([
    ('convection', 0, 1, 0, 'np.sin(10*x)', 200, 200, 0),
    ('convection', 0, 1, 0, 'np.exp(-np.power((x - 0.5*np.pi)/(np.pi/64), 2.)/2.)', 200, 200, 0),
])

x = dataset[0]['u'].squeeze(-1)
print(x.shape)
u, s, vh = linalg.svd(x)

print(u.shape, s.shape, vh.shape)

# print(s[..., 0])
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

plt.title('modes')
plt.plot(np.abs(s))
plt.show()