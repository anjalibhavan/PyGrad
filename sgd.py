
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import numpy as np

# Load dataset and split into train and test sets

diabetes=load_diabetes()
Xdata = diabetes.data
ydata = diabetes.target
x_train,x_test,y_train,y_test=train_test_split(Xdata,ydata,random_state=42)

# Initialize training parameters

n_iterations=100
thetaa=np.zeros(x_train.shape[1])
alpha=0.5

# Stochastic Gradient Descent

def SGD(X,theta,y):
    X_trans=X.transpose()
    gradient=0
    for i in range(1,n_iterations):
        X_new=shuffle(X)
        for q in range(0,X.shape[0]):
            p=np.random.randint(0,X_new.shape[0])
            h=np.dot(X_new[p],theta)
            loss=h-y[p]
            gradient=np.dot(X_trans[:,p],loss)
            theta=theta-alpha*gradient
    return theta

# Calculate new weights and print them

new_theta=SGD(x_train,thetaa,y_train)
print(new_theta)
