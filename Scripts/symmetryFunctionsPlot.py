import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

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
           
def G4angular(theta, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta
           
           
def G5(Rij, Rik, cosTheta, width, cutoff, thetaRange, inversion):
    
    return 2**(1-thetaRange) * (1 + inversion*cosTheta)**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff)

# set parameters
plt.rc('lines', linewidth=1.5)
plt.rc('axes', prop_cycle=(cycler('color', ['g', 'k', 'y', 'b', 'r', 'c', 'm']) ))
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=25)


##### fc plot #####
"""Rij = np.linspace(0, 1, 1000)
cutoff = 1.0
plt.plot(Rij, cutoffFunction(Rij, cutoff))
plt.xlabel(r'$R/R_c$', fontsize=15)
plt.ylabel(r'$f_c$', fontsize=15)
#plt.savefig('../Figures/Theory/cutoffFunction.pdf')
plt.show()"""



##### G1 plot #####
"""Rij = np.linspace(0, 14, 1000)

legends = []
for cutoff in [3.0, 5.0, 7.0, 9.0, 11.0]:
    functionValue = G1(Rij, cutoff)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$R_c = %3.1f \, \mathrm{\AA{}}$' % cutoff)
    plt.hold('on')
    
plt.legend(legends, prop={'size':20})
plt.ylabel(r'$G_1$')
plt.tight_layout()
#plt.show()
plt.savefig('../Figures/Theory/G1.pdf')

##### G2 plot #####

plt.figure()

Rs = 0.0
cutoff = 11.0
legends = []
for eta in [5.0, 0.9, 0.3, 0.15, 0.07, 0.03, 0.01]:
    functionValue = G2(Rij, eta, cutoff,  Rs)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$\eta = %3.2f \, \mathrm{\AA{}}^{-2}$' % eta)
    plt.hold('on')

plt.legend(legends, prop={'size':20})
plt.ylabel(r'$G_2, \, R_s = 0 \, \mathrm{\AA{}}$')
plt.tight_layout()
#plt.show()
plt.savefig('../Figures/Theory/G2_1.pdf')

plt.figure()

eta = 3.0
cutoff = 11.0
legends = []
for Rs in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    functionValue = G2(Rij, eta, cutoff,  Rs)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$R_s = %.1f \, \mathrm{\AA{}}$' % Rs)
    plt.hold('on')

plt.legend(legends, prop={'size':20})
plt.xlabel(r'$R_{ij} [\mathrm{\AA{}}]$')
plt.ylabel(r'$G_2, \, \eta=3.0 \, \mathrm{\AA{}}^{-2}$')
plt.tight_layout()
plt.savefig('../Figures/Theory/G2_2.pdf')

plt.figure()

##### G3 plot #####

cutoff = 11.0
legends = []
for kappa in [0.5, 1.0, 1.5, 2.0]:
    functionValue = G3(Rij, cutoff, kappa)
    functionValue[np.where(Rij > cutoff)[0]] = 0
    plt.plot(Rij, functionValue)
    legends.append(r'$\kappa = %.1f \, \mathrm{\AA{}}^{-1}$' % kappa)
    plt.hold('on')

plt.plot(Rij, np.zeros(1000), 'k--')
plt.legend(legends, prop={'size':20})
plt.xlabel(r'$R_{ij} [\mathrm{\AA{}}]$')
plt.ylabel(r'$G_3$')
plt.tight_layout()
#plt.show()
plt.savefig('../Figures/Theory/G3.pdf')

#plt.savefig('../Figures/Theory/radialSymmFuncs.pdf', dpi=1000)"""


##### G4 plot #####

plt.figure()

theta = np.linspace(0, 2*np.pi, 1000) 

inversion = 1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G4angular(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')
    
plt.legend(legends, prop={'size':20}, loc=9)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$G^4/G^5$ angular part')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
#plt.show()
plt.savefig('../Figures/Theory/G4G5angular1.pdf')
    
plt.figure()    
    
inversion = -1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G4angular(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')

plt.legend(legends, fontsize=25, prop={'size':18}, loc=1)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$G^4/G^5$ angular part')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
#plt.show()
plt.savefig('../Figures/Theory/G4G5angular2.pdf')


##### G5 plot #####







            

            



