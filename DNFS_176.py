from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

f1 = plt.figure(1)
plt.imshow(X_train[0])
f2 = plt.figure(2)
plt.imshow(X_train[1])
plt.show()

#check image shape and data count
print(X_train[0].shape, len(X_train))
print(X_train[0].shape, len(X_test))

#reshape data to fit model
X_train = X_train.reshape(len(X_train),28,28,1)
X_test = X_test.reshape(len(X_test),28,28,1)

#One-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]

#Create model
model = Sequential()

#Add Input CNN Layer
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

#Add second CNN Layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#Add the fully connected layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 6 images in the test set
model.predict(X_test[:6])

#actual results for first 6 images in the test set
y_test[:6]