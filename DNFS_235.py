from functools import partial
import numpy as np
import FuzzyART as f
import sklearn.datasets as ds


#Used for regularization so that we can penalize the parameters that are not important
l1_norm = partial(np.linalg.norm, ord=1, axis=-1)

if __name__ == '__main__':

    #load the dataset in the python environment
    iris = ds.load_iris()

    #standardize the dataset
    data = iris['data'] / np.max(iris['data'], axis=0)

    #Initialize the FuzzyART Hyperparameters
    net = f.FuzzyART('trainingData.txt', 'testingData.txt')

    #Train on the data
    net.train(data, epochs=100)

    #Print the Cluster Results
    print(net.test(data).astype(int))

    #Match the cluster results
    print(iris['target'])
