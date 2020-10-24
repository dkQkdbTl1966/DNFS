#Importing Necessary Package
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
%matplotlib

#defining thr Numpy array for Tip Quality
x_qual=np.arange(0,11,1)

#defining the Numpy array for Triangular membership functions
qual_lo=sk.trimf(x_qual,[0,0,5])