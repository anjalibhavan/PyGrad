from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset and split into train and test sets

diabetes = load_diabetes()
Xdata = diabetes.data
ydata = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(Xdata,ydata,random_state=42)

# To add a dataset of your own, uncomment the following lines and comment out the previous four lines:
'''
import pandas as pd
cancer=pd.read_csv('path of csv file')
Xdata=cancer.iloc[:,:] #set the indices as per the column orientation in the csv file
ydata=cancer.iloc[:,:] #set the indices as per the column orientation in the csv file
x_train,x_test,y_train,y_test = train_test_split(Xdata,ydata,random_state=42)
'''

# Initialize training parameters

thetaa = np.zeros(x_train.shape[1])
alpha = 0.5
batch_size=10

# Function for dividing the data into batches

def getbatches(X,y,batch_size,i):
    X_new=X[i:i+batch_size,:]
    y_new=y[i:i+batch_size]    
    return X_new,y_new


# Mini-Batch Gradient Descent

def minibatch(X,theta,y,batch_size):
    num_batches=int(X.shape[0]/batch_size)    
    gradient=0
    for i in range(0,num_batches):
        X_batch,y_batch=getbatches(X,y,batch_size,i)
        h=np.dot(X_batch,theta)
        loss=h-y_batch
        X_trans=X_batch.transpose()
        gradient=np.dot(X_trans,loss)/batch_size
        theta=theta-alpha*gradient
    return theta


# Calculate new weights and print them
new_theta = SGD(x_train,thetaa,y_train,batch_size)
print(new_theta)
