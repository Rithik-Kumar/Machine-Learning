import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Salary_Data1.csv')
#print(dataset.head())
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.show()

#col = ['r','g']
#plt.scatter(y_test,y_pred, color=col)
#plt.show()
