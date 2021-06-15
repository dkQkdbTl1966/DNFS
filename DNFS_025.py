import numpy as np
import skfuzzy as fuzz
#Defining the Numpy array for Tip Quality
x_qual = np.arange(0, 11, 1)
#Defining the Numpy array for Triangular membership functions
qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
