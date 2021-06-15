import sklearn
import skfuzzy
import fuzzycmeans

import pandas as pd
import numpy as np

import math
import random
import logging

#
from fuzzycmeans.fuzzy_clustering import fcm
# from fuzzycmeans.visualization import draw_model_2d
from fuzzycmeans.visualization import draw_model_2d
#
from sklearn import preprocessing

# Importing theairlines data
dataset = pd.read_csv("AirlinesCluster.csv")

print (dataset)


# Making a copy so that original data remains unaffected
dataset1 = dataset.copy()

# Selecting only first 500 rows for faster computation
dataset1 = dataset1[["Balance", "BonusMiles"]][:500]
dataset1_standardized = preprocessing.scale(dataset1)

# Standardizing the data to scale it between the upper and lower limit of 1 and 0
dataset1_standardized = pd.DataFrame(dataset1_standardized)

print (dataset1_standardized)

# Telling the package class to stop the unnecessary output
FCM=fcm()
FCM.set_logger(tostdout=False)

# Defining k=5
fcm = fcm(n_clusters=5)

# Training on data
fcm.fit(dataset1_standardized)

# Testing on same data
predicted_membership = fcm.predict(np.array(dataset1_standardized))

# Visualizing the data
draw_model_2d(fcm, data=np.array(dataset1_standardized), membership=predicted_membership)
