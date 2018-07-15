
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Loading breast cancer (Wisconsin) in-built dataset. 

cancer = load_breast_cancer()
Xdata = cancer.data
ydata = cancer.target
x_train, x_test, y_train, y_test = train_test_split(Xdata,ydata,random_state=42)

# To add a dataset of your own, uncomment the following lines and comment out the previous four lines:
'''
import pandas as pd
cancer=pd.read_csv('path of csv file')
Xdata=cancer.iloc[:,:] #set the indices as per the column orientation in the csv file
ydata=cancer.iloc[:,:] #set the indices as per the column orientation in the csv file
x_train,x_test,y_train,y_test = train_test_split(Xdata,ydata,random_state=42)
'''

# Initializing weights, number of epochs and learning rate for training

thetaa = np.zeros(x_train.shape[1])
alpha = 0.001
num_epochs = 1000

# Implementing the sigmoid function for calculating the hypotheses

def sigmoid(a):
    sigma = 1/(1+np.exp(-a))
    return sigma

# Main function for Logistic Regression

def logisticregression(X,theta,y):
    Xtrans = X.transpose()
    grad = 0
    for i in range(0,num_epochs):
        h = sigmoid(np.dot(X,theta))
        loss = h-y
        gradient = np.dot(Xtrans,loss)
        theta = theta-alpha*gradient
    return theta


# Calculate weights of data for prediction
newtheta = logisticregression(x_train,thetaa,y_train)

# Predictions using the above calculated weights
selfpreds = np.round(sigmoid(np.dot(x_test,newtheta)))

# Comparison with scikit-learn's Logistic Regression classifier
model = LogisticRegression()
model.fit(x_train,y_train)
modelpred = model.predict(x_test)

# Final results
print('Accuracy using scikit-learn model',accuracy_score(y_test,modelpred))
print('Accuracy using self implemented model',accuracy_score(y_test,selfpreds))

    
