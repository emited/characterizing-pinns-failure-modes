import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,  2 * np.pi, 1000)
y = np.sin(3*np.sin(3*np.sin(np.sin(2*x+1))+1) + 2)
y = np.sin(3*np.sin(3*np.sin(3.5*np.sin(x))))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.plot(np.abs(np.fft.fft(y)))
plt.show()