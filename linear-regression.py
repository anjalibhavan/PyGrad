alpha=0.5
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
thetaa=np.zeros(10)

diabetes=load_diabetes()
Xdata = diabetes.data
ydata = diabetes.target
x_train,x_test,y_train,y_test=train_test_split(Xdata,ydata,random_state=42)

def linreg(X,theta,y):
    Xtrans=X.transpose()
    grad=0
    for i in range(1,10000):
        h=np.dot(X,theta)
        loss=h-y
        
        grad=np.dot(Xtrans,loss)
        theta=theta-alpha*grad
    return theta


model=LinearRegression()
model.fit(x_train,y_train)
modelpred=model.predict(x_test)

newtheta=linreg(x_train,thetaa,y_train)
preds=(np.dot(x_test,newtheta))

print('self accuracy',mean_squared_error(y_test,preds))
print('model accuracy',mean_squared_error(y_test,modelpred))

    
