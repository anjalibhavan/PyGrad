def batch_gradient_descent(X,y,theta=np.empty(),learning_rate=0.1,num_epochs=100):
    X_trans=X.transpose()
    gradient=0
    for i in range(0,num_epochs):
        hypothesis=np.dot(X,theta)
        loss=hypothesis-y
        gradient=np.dot(X_trans,loss)
        theta=theta-learning_rate*gradient
    return theta
