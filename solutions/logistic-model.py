def cross_entropy(X, Y, a):
    Z = logodds(X, a)
    logliks = -Z * (1 - Y) - np.log(1 + np.exp(-Z))
    return -logliks.mean()




def gradient_descent(X, Y, a, η=0.01):
    nsamples = Y.shape[0]
    
    Z = X @ a
    Yhat = 1 / (1 + np.exp(-Z))
    δ = Yhat - Y
    dJda = X.T @ δ / nsamples
    assert dJda.shape == a.shape
    return a - η * dJda