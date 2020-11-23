import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
 
# convert an array of values into a dataset matrix
def create_dataset(dataset, step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-step-1):
    	a = dataset[i:(i+step), 0]
    	dataX.append(a)
    	dataY.append(dataset[i + step, 0])
	return numpy.array(dataX), numpy.array(dataY)
    
# load the dataset
dataframe = pd.read_csv('carriage.csv', usecols=[1])
dataset = dataframe.values
dataset = dataset.astype('float32')
 
# standardize the dataset
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
 
# split into train and test sets
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
 
# Reshaping Data for the model
step = 1
train_X, train_Y = create_dataset(train, step)
test_X, test_Y = create_dataset(test, step)
 
train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = numpy.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
 
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(1, step)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(train_X, train_Y, epochs=10, batch_size=50, verbose=1)

# create and fit the GRU network
model1 = Sequential()
model1.add(GRU(10, input_shape=(1, step)))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.summary()
model1.fit(train_X, train_Y, epochs=10, batch_size=50, verbose=1)
 
# make predictions from LSTM
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

# make predictions from GRU
trainPredict1 = model1.predict(train_X)
testPredict1 = model1.predict(test_X)
 
# invert predictions from LSTM
trainPredict = scaler.inverse_transform(trainPredict)
train_Y = scaler.inverse_transform([train_Y])
testPredict = scaler.inverse_transform(testPredict)
test_Y = scaler.inverse_transform([test_Y])

# invert predictions from GRU
trainPredict1 = scaler.inverse_transform(trainPredict1)
testPredict1 = scaler.inverse_transform(testPredict1)
 
# calculate root mean squared error for LSTM
print("*****Results for LSTMs*****")
trainScore = math.sqrt(mean_squared_error(train_Y[0], trainPredict[:,0]))
print('Error in Training data is: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], testPredict[:,0]))
print('Error in Testing data is: %.2f RMSE' % (testScore))

# calculate root mean squared error for GRU
print("*****Results for GRUs*****")
trainScore1 = math.sqrt(mean_squared_error(train_Y[0], trainPredict1[:,0]))
print('Error in Training data is: %.2f RMSE' % (trainScore1))
testScore1 = math.sqrt(mean_squared_error(test_Y[0], testPredict1[:,0]))
print('Error in Testing data is: %.2f RMSE' % (testScore1))