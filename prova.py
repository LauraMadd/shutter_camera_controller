from scipy.signal import sawtooth, square, triang
import matplotlib.pyplot as plt
import numpy as np




num_samples=10**5
frequency_im=0.0498
t = np.linspace(0, 1/frequency_im, num_samples, dtype=np.float16)
signal_2 = 5 * (square(2 * np.pi * t  * frequency_im, duty = .1)+1)/2.
signal_2[int(len(signal_2)*3/4+1):] = 0


plt.figure('plot')

plt.plot(t,signal_2,'b-')
plt.show()