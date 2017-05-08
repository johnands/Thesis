"""
Script that calculates how the number of terms of G4/G5
scales with average number of neighbours
"""

import numpy as np
import matplotlib.pyplot as plt

neighMax = 100
nNeighs = np.arange(1, neighMax+1, 1)
nTerms = np.zeros(neighMax)
for i in nNeighs:
    s = 0
    for j in xrange(i):
        s += j
    nTerms[i-1] = s

coeffs = np.polyfit(nNeighs, nTerms, 2)
poly = coeffs[0]*nNeighs**2 + coeffs[1]*nNeighs

plt.plot(nNeighs, nTerms, 'b-', nNeighs, poly, 'g-')
plt.show()

polynomial = '%.1fx**2 + %.1fx' % (coeffs[0], coeffs[1])
print polynomial

print np.polyval(coeffs, 80) / np.polyval(coeffs, 40)


