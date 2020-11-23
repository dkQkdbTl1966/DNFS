from functools import partial
import numpy as np
import FuzzyART as f
import sklearn.datasets as ds

l1_norm = partial(np.linalg.norm, ord=1, axis=-1)#Used for regularization so that we can penalize the parameters that are not important

if __name__ == '__main__': 

    iris = ds.load_iris()#load the dataset in the python environment

data = iris['data'] / np.max(iris['data'], axis=0)#standardize the dataset

net = f.FuzzyART(alpha=0.5, rho=0.5) #Initialize the FuzzyART Hyperparameters

    net.train(data, epochs=100) #Train on the data

    print(net.test(data).astype(int)) #Print the Cluster Results

    print(iris['target']) #Match the cluster results