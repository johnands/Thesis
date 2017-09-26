# plot different activation functions for visualization

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

points = np.linspace(-5, 5, 5000)
hyperbolicTan = np.tanh(points)
sigmoid = 1./(1 + np.exp(-points))

plt.plot(points, sigmoid, 'g-')
plt.hold('on')
plt.plot(points, hyperbolicTan, 'b--')
plt.grid('on')
plt.axis([-5, 5, -1, 1])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['sigmoid', 'tanh'], loc=2, prop={'size':20})
plt.tight_layout()
#plt.savefig('../../Figures/Theory/activationFunctionsAltered.pdf')
#plt.show()

def rectifier(x):
    if x > 0:
        return x
    else:
        return 0
       
relu = np.zeros(len(points))
for i, x in enumerate(points):
    relu[i] = rectifier(x)
    
plt.figure()
plt.plot(points, relu)
plt.axis([-5, 5, -1, np.max(relu)])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid('on')
plt.legend(['Rectifier'], loc=2, prop={'size':15})
plt.tight_layout()
#plt.savefig('../../Figures/Theory/reluActivation.pdf')
#plt.show()



