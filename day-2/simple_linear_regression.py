from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../dataset/studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/4, random_state=0)

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
