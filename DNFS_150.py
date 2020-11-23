#Reading Data
import pandas as pd
house_price = pd.read_csv("hp_train.csv")

#Partition into Categorical and Numerical Variables
import numpy as np
cat = house_price.select_dtypes(include=[object])
num = house_price.select_dtypes(include=[np.number])

#Checking Null Values
cat.isnull().sum()
num.isnull().sum()

#Removing unnecessary columns
cat.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=True)

#Removing Categorical Null Values with Mode
cat.BsmtCond.fillna(cat.BsmtCond.value_counts().idxmax(),inplace=True)
cat.BsmtQual.fillna(cat.BsmtQual.value_counts().idxmax(),inplace=True)
cat.BsmtExposure.fillna(cat.BsmtExposure.value_counts().idxmax(),inplace=True)
cat.BsmtFinType1.fillna(cat.BsmtFinType1.value_counts().idxmax(),inplace=True)
cat.BsmtFinType2.fillna(cat.BsmtFinType2.value_counts().idxmax(),inplace=True)
cat.FireplaceQu.fillna(cat.FireplaceQu.value_counts().idxmax(),inplace=True)
cat.GarageCond.fillna(cat.GarageCond.value_counts().idxmax(),inplace=True)
cat.GarageFinish.fillna(cat.GarageFinish.value_counts().idxmax(),inplace=True)
cat.GarageQual.fillna(cat.GarageQual.value_counts().idxmax(),inplace=True)
cat.GarageType.fillna(cat.GarageType.value_counts().idxmax(),inplace=True)
cat.Electrical.fillna(cat.Electrical.value_counts().idxmax(),inplace=True)
cat.MasVnrType.fillna(cat.MasVnrType.value_counts().idxmax(),inplace=True)

#Removing Numerical Null Values with Mean
num.LotFrontage.fillna(num.LotFrontage.mean(),inplace=True)
num.GarageYrBlt.fillna(num.GarageYrBlt.mean(),inplace=True)
num.MasVnrArea.fillna(num.MasVnrArea.mean(),inplace=True)

#Converting words to Integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat1 = cat.apply(le.fit_transform)

#Combining two dataframes
house_price2 = pd.concat([cat1, num], axis=1)

#Getting Dependent and Independent Variables
X = house_price2.drop(["SalePrice"], axis=1)
Y = pd.DataFrame(house_price2["SalePrice"])

#Getting Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Applying Linear Regression
import statsmodels.api as sm
est = sm.OLS(Y_train, X_train)
est2 = est.fit()
est2.summary()