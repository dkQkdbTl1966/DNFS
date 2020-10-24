#Importing Necessary Package
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
#%matplotlib inline

#defining the Fuzzy Range from a speed of 30 to 90
x=np.arange(30,90,0.1)

#defining the s-shaped membership functions
slow=fuzz.smf(x,60,80)
medium=fuzz.smf(x,50,60)
medium_fast=fuzz.smf(x,30,50)
full_speed=fuzz.smf(x,20,30)

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