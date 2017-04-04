# plot different activation functions for visualization

import numpy as np
import matplotlib.pyplot as plt

points = np.linspace(-4, 4, 500)
hyperbolicTan = np.tanh(points)
sigmoid = 1./(1 + np.exp(-points))

plt.plot(points, sigmoid, 'g-')
plt.hold('on')
plt.plot(points, hyperbolicTan, 'b--')
plt.grid('on')
plt.legend(['Sigmoid', 'Hyperbolic tangent'], loc=2)
plt.show()

