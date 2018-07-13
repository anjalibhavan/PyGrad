from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Loading breast cancer (Wisconsin) in-built dataset. 

diabetes=load_diabetes()
Xdata = diabetes.data
ydata = diabetes.target
x_train,x_test,y_train,y_test=train_test_split(Xdata,ydata,random_state=42)

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
num_epochs=1000

# Main function for Linear Regression

def linearregression(X,theta,y):
    Xtrans=X.transpose()
    grad=0
    for i in range(0,num_epochs):
        h=np.dot(X,theta)
        loss=h-y
        grad=np.dot(Xtrans,loss)
        theta=theta-alpha*grad
    return theta


# Calculate weights of data for prediction
newtheta=linearregression(x_train,thetaa,y_train)

# Predictions using the above calculated weights
preds=(np.dot(x_test,newtheta))

# Comparison with scikit-learn's Linear Regressor

model=LinearRegression()
model.fit(x_train,y_train)
modelpred=model.predict(x_test)

# Final results
print('self mean squared error',mean_squared_error(y_test,preds))
print('model mean squared error',mean_squared_error(y_test,modelpred))

    
