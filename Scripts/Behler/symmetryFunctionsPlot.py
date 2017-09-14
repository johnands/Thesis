import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentdir = os.path.dirname(parentdir)
grandGrandParentdir = os.path.dirname(grandParentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentdir)
sys.path.insert(2, grandGrandParentdir)
import TensorFlow.Tools.matplotlibParameters

def cutoffFunction(rVector, cutoff, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*rVector / cutoff) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        value[np.where(rVector > cutoff)[0]] = 0
        
    return value
 
    
def G1(Rij, cutoff):
    
    return cutoffFunction(Rij, cutoff)
    
    
def G2(Rij, width, cutoff, center):
    
    return np.exp(-width*(Rij - center)**2) * cutoffFunction(Rij, cutoff)
    
    
def G3(Rij, cutoff, kappa):
    
    return np.cos(kappa*Rij) * cutoffFunction(Rij, cutoff)
    
    
def G4(Rij, Rik, Rjk, theta, width, cutoff, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta * \
           np.exp( -width*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) * cutoffFunction(Rjk, cutoff, cut=True)
           
def G4G5angular(theta, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta
           
           
def G5(Rij, Rik, cosTheta, width, cutoff, thetaRange, inversion):
    
    return 2**(1-thetaRange) * (1 + inversion*cosTheta)**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff)

# set parameters
#plt.rc('lines', linewidth=1.5)
#plt.rc('axes', prop_cycle=(cycler('color', ['g', 'k', 'y', 'b', 'r', 'c', 'm']) ))
#plt.rc('xtick', labelsize=20)
#plt.rc('ytick', labelsize=20)
#plt.rc('axes', labelsize=25)

# change parameters
"""plt.rc('axes', labelsize=20)


##### fc plot #####
Rij = np.linspace(0, 1, 1000)
cutoff = 1.0
plt.plot(Rij, cutoffFunction(Rij, cutoff))
plt.xlabel(r'$r_{ij}/r_c$')
plt.ylabel(r'$f_c(r_{ij}/r_c)$')
plt.tight_layout()
plt.savefig('../../Figures/Theory/cutoffFunction.pdf')
#plt.show()"""


# change parameters
"""plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('axes', labelsize=27)



##### G1 plot #####
Rij = np.linspace(0, 14, 1000)

legends = []
for cutoff in [3.0, 5.0, 7.0, 9.0, 11.0]:
    functionValue = G1(Rij, cutoff)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$r_c = %3.1f$' % cutoff)
    plt.hold('on')
    
plt.legend(legends, prop={'size':24})
plt.ylabel(r'$G_i^1$')
plt.tight_layout()
#plt.show()
plt.savefig('../../Figures/Theory/G1.pdf')

##### G2 plot #####

plt.figure()

Rs = 0.0
cutoff = 11.0
legends = []
for eta in [5.0, 0.9, 0.3, 0.15, 0.07, 0.03, 0.01]:
    functionValue = G2(Rij, eta, cutoff,  Rs)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$\eta = %3.2f$' % eta)
    plt.hold('on')

plt.legend(legends, prop={'size':24})
plt.ylabel(r'$G_i^2, \; r_s = 0$')
plt.tight_layout()
#plt.show()
plt.savefig('../../Figures/Theory/G2_1.pdf')

plt.figure()

eta = 3.0
cutoff = 11.0
legends = []
for Rs in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    functionValue = G2(Rij, eta, cutoff,  Rs)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$r_s = %.1f$' % Rs)
    plt.hold('on')

plt.legend(legends, prop={'size':24})
plt.xlabel(r'$r_{ij}$')
plt.ylabel(r'$G_i^2, \; \eta=3.0$')
plt.tight_layout()
plt.savefig('../../Figures/Theory/G2_2.pdf')

plt.figure()

##### G3 plot #####

cutoff = 11.0
legends = []
for kappa in [0.5, 1.0, 1.5, 2.0]:
    functionValue = G3(Rij, cutoff, kappa)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$\kappa = %.1f$' % kappa)
    plt.hold('on')

plt.plot(Rij, np.zeros(1000), 'k--')
plt.legend(legends, prop={'size':22})
plt.xlabel(r'$r_{ij}$')
plt.ylabel(r'$G_i^3$')
plt.tight_layout()
#plt.show()
plt.savefig('../../Figures/Theory/G3.pdf')

#plt.savefig('../Figures/Theory/radialSymmFuncs.pdf', dpi=1000)"""


##### G4 plot #####

plt.figure()

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('axes', labelsize=27)

theta = np.linspace(0, 2*np.pi, 1000) 

inversion = 1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G4G5angular(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')
    
plt.legend(legends, prop={'size':20}, loc=9)
plt.xlabel(r'$\theta_{jik}$')
plt.ylabel(r'$G_i^\theta$')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
#plt.show()
#plt.savefig('../../Figures/Theory/G4G5angular1.pdf')
    
plt.figure()    
    
inversion = -1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G4G5angular(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')

plt.legend(legends, fontsize=25, prop={'size':18}, loc=1)
plt.xlabel(r'$\theta_{jik}$')
#plt.ylabel(r'$G_i^4/G_i^5$ angular part')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
#plt.show()
#plt.savefig('../../Figures/Theory/G4G5angular2.pdf')


##### G5 plot #####







            

            



