
# Stochastic Gradient Descent

def StochasticGradientDescent(X,y,theta,n_epochs,learning_rate):
    X_trans = X.transpose()
    gradient = 0
    for i in range(1,n_epochs):
        X_new = shuffle(X)
        for q in range(0,X.shape[0]):
            p = np.random.randint(0,X_new.shape[0])
            hypothesis = np.dot(X_new[p],theta)
            loss = hypothesis-y[p]
            gradient = np.dot(X_trans[:,p],loss)
            theta = theta-learning_rate * gradient
    return theta

