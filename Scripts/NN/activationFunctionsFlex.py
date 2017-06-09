# visualize flexibility of tanh activation function

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

points = np.linspace(-3, 3, 5000)
function = np.tanh

def activation(function, x, c1, c2, c3, c4):
    
    return c1*function(c2*x + c3) + c4
    

fig = plt.figure()

# c1: scaling the function values
ax = fig.add_subplot(2,2,1)
legends = []
for c1 in [0.5, 0.75, 1.0, 1.25, 1.5]:
    values = activation(function, points, c1, 1.0, 0, 0)
    legends.append(r'$c_1 = %1.2f$' % c1)
    ax.text(0.1, 0.8, 'a)', fontsize=20,
            #horizontalalignment='left',
            transform=ax.transAxes)
    plt.plot(points, values)
#plt.legend(legends, loc=2)
    
ax = fig.add_subplot(2,2,2)
legends = []
for c2 in [0.5, 0.75, 1.0, 1.25, 1.5]:
    values = activation(function, points, 1.0, c2, 0, 0)
    legends.append(r'$c_2 = %1.2f$' % c2)
    ax.text(0.1, 0.8, 'b)', fontsize=20,
            #horizontalalignment='left',
            transform=ax.transAxes)
    plt.plot(points, values)
#plt.legend(legends, loc=2)

ax = fig.add_subplot(2,2,3)
legends = []
for c3 in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    values = activation(function, points, 1.0, 1.0, c3, 0)
    legends.append(r'$c_3 = %2.2f$' % c3)
    ax.text(0.1, 0.8, 'c)', fontsize=20,
            #horizontalalignment='left',
            transform=ax.transAxes)
    plt.plot(points, values)
#plt.legend(legends, loc=2)

ax = fig.add_subplot(2,2,4)
legends = []
for c4 in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    values = activation(function, points, 1.0, 1.0, 0, c4)
    legends.append(r'$c_4 = %2.2f$' % c4)
    ax.text(0.1, 0.8, 'd)', fontsize=20,
            #horizontalalignment='left',
            transform=ax.transAxes)
    plt.plot(points, values)
#plt.legend(legends, loc=2)

plt.tight_layout()
plt.savefig('../../Figures/Theory/activationFlex.pdf')
#plt.show()





