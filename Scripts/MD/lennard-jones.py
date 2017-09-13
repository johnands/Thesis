import numpy as np
import matplotlib.pyplot as plt

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentdir = os.path.dirname(parentdir)
grandGrandParentdir = os.path.dirname(grandParentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentdir)
sys.path.insert(2, grandGrandParentdir)
import TensorFlow.Tools.matplotlibParameters

r = np.linspace(0.8, 4, 5000)
epsilon = 1
sigma = 1
LJ = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
epsilon = 1.5
sigma = 2
LJ2 = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r, LJ, 'b-', r, LJ2, 'g-')
plt.xlabel(r'$r_{ij}  \; [\mathrm{\AA{}}]$')
plt.ylabel(r'$V_{\mathrm{LJ}}(r_{ij}) \; [\mathrm{eV}]$')
plt.axis([0.8, 4, -2, 3])
plt.legend([r'$\epsilon=1.0 \; \mathrm{eV}, \; \sigma=1.0 \; \mathrm{\AA{}}$', r'$\epsilon = 1.5 \; \mathrm{eV}, \; \sigma=2.0 \; \mathrm{\AA{}}$'], prop={'size':17})
#plt.show()
plt.tight_layout()

# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')

plt.savefig('../../Figures/Theory/LJ.pdf')
