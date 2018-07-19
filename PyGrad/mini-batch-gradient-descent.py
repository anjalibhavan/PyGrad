
# Function for dividing the data into batches

def get_batches(X,y,batch_size,i):
    X_new = X[i:i+batch_size,:]
    y_new = y[i:i+batch_size]    
    return X_new, y_new


# Mini-Batch Gradient Descent

def MiniBatchGradientDescent(X,y,theta,learning_rate=0.1,batch_size=10):
    num_batches = int(X.shape[0]/batch_size)    
    gradient = 0
    for i in range(0,num_batches):
        X_batch, y_batch = get_batches(X,y,batch_size,i)
        hypothesis = np.dot(X_batch,theta)
        loss = hypothesis - y_batch
        X_trans = X_batch.transpose()
        gradient = np.dot(X_trans,loss)/batch_size
        theta = theta - alpha * gradient
    return theta
