import numpy as np
import matplotlib.pyplot as plt

# parameters                            
A = 7.049556277
B = 0.6022245584
p = 4.0
q = 0.0
a = 1.80
Lambda = 21.0
gamma = 1.20
cosC = -1.0/3
epsilon = 1.0
sigma = 2.0951

cut = a*sigma

Rij 	= np.linspace(1.5, cut-0.001, 500)
Rik 	= 2.5
theta	= 109*(np.pi/180)#np.linspace(0, 2*np.pi, 500)

# Stillinger-Weber            
stillingerWeber = epsilon*A*(B*(sigma/Rij)**p - (sigma/Rij)**q) * \
                  np.exp(sigma / (Rij - a*sigma)) + \
				  epsilon*Lambda*(np.cos(theta) - cosC)**2 * \
				  np.exp( (gamma*sigma) / (Rij - a*sigma) ) * \
				  np.exp( (gamma*sigma) / (Rik - a*sigma) )
				  
swTwoBody = epsilon*A*(B*(sigma/Rij)**p - (sigma/Rij)**q) * \
            np.exp(sigma / (Rij - a*sigma))
            
swThreeBody = epsilon*Lambda*(np.cos(theta) - cosC)**2 * \
			  np.exp( (gamma*sigma) / (Rij - a*sigma) ) * \
			  np.exp( (gamma*sigma) / (Rik - a*sigma) )
                  
#plt.plot(Rij, stillingerWeber)
#plt.show()

#plt.plot(Rij, swTwoBody)
#plt.show()

#plt.plot(theta*180/np.pi, swThreeBody)
#plt.show()

plt.plot(Rij, swThreeBody)
plt.show()
                    

