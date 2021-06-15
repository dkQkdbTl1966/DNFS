#Reading data
import pandas as pd

data = pd.read_csv("~/cy_code/train_tt.csv")

#Splitting Data into Categorical and Numerical Dataframes
import numpy as np

data_cat = data.select_dtypes(include=[object])
data_num = data.select_dtypes(include=np.number)

#Checking the number of null values
print (data_cat.isnull().sum())
print (data_num.isnull().sum())

#Dropping the Columns having null values and columns which are not important
data_cat.drop(["Cabin","Embarked","Name","Ticket"], axis=1, inplace=True)
data_num.drop(["Age","PassengerId"], axis=1, inplace=True)

#Checking the null values again
print (data_cat.isnull().sum())
print (data_num.isnull().sum())

#Converting categorical variables into numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_cat = data_cat.apply(le.fit_transform)

#Combining both dataframes
data = pd.concat([data_cat,data_num], axis=1)

#Defining dependent and independent variables
X = data.drop(["Survived"], axis=1)
Y = pd.DataFrame(data[["Survived"]])

#Defining data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)

#Predicting Values
pred = lr.predict(X_test)

#Finding different classification measures
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

print (confusion_matrix(pred,y_test))

print (accuracy_score(pred, y_test))

print (recall_score(pred, y_test))

print (precision_score(pred, y_test))

print (f1_score(pred, y_test))

from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot

# predict probabilities
probs = lr.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')

# show the plot
pyplot.show()
