
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

thetaa=np.zeros(30)
alpha=0.001

cancer=load_breast_cancer()
Xdata = cancer.data
ydata = cancer.target
x_train,x_test,y_train,y_test=train_test_split(Xdata,ydata,random_state=42)


def sigmoid(a):
    sigma=1/(1+np.exp(-a))
    return sigma

def logisticregression(X,theta,y):
    Xtrans=X.transpose()
    grad=0
    for i in range(1,5000):
        h=sigmoid(np.dot(X,theta))
        loss=h-y
        gradient=np.dot(Xtrans,loss)
        theta=theta-alpha*gradient
    return theta

newtheta=logisticregression(x_train,thetaa,y_train)

preds=np.round(sigmoid(np.dot(x_test,newtheta)))

model=LogisticRegression()
model.fit(x_train,y_train)
ypred=model.predict(x_test)

print('logacc',accuracy_score(y_test,ypred))
print('self accuracy',accuracy_score(y_test,preds))

    
