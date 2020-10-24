#Importing Necessary Package
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
#%matplotlib inline

#defining the Fuzzy Range from a speed of 30 to 90
x=np.arange(30,90,0.1)

#defining the Gaussian membership functions
slow=fuzz.gaussmf(x,80,4)
medium=fuzz.gaussmf(x,60,4)
medium_fast=fuzz.gaussmf(x,50,4)
full_speed=fuzz.gaussmf(x,30,4)

#plotting the Membership Functions defined
plt.figure()
plt.plot(x,full_speed,'b', linewidth=1.5, label='Full Speed')
plt.plot(x,medium_fast,'k', linewidth=1.5, label='Medium Fast')
plt.plot(x,medium,'m', linewidth=1.5, label='Medium Powered')
plt.plot(x,slow,'r', linewidth=1.5, label='slow')
plt.title('Penalty Kick Fuzzy')
plt.ylabel('Membership')
plt.xlabel('Speed(miles/hr)')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1, fancybox=True, shadow=True)
plt.show()