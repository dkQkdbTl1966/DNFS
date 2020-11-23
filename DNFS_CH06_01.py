# Importing necessary libraries
import anfis
from anfis.membership import membershipfunction, mfDerivs
import numpy
training_data = numpy.loadtxt("training.txt", usecols=[1,2,3])
X = training_data [:,0:2]
Y = training_data [:,2]
# Defining the Membership Functions
mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]], [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]
# Updating the model with Membership Functions
mfc = membershipfunction.MemFuncs(mf)
# Creating the ANFIS Model Object
anf = anfis.ANFIS(X, Y, mfc)
# Fitting the ANFIS Model
anf.trainHybridJangOffLine(epochs=20)
# Printing Output
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
# Plotting Model Performance
anf.plotErrors()
anf.plotResults()