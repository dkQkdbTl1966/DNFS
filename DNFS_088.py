import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#Defining the T-Norm Function
def t_norm(mfx, mfy):
    tnorm=np.fmin(mfx, mfy)
    return tnorm

#defining the Fuzzy Range from a speed of 30 to 90
x=np.arange(30,90,0.1)

#Defining sigmoidal membership function
full_speed=fuzz.sigmf(x, 80, 2)
slow=fuzz.sigmf(x, 30, 2)

#Finding the Intersection
t_norm(full_speed, slow)

plt.show()