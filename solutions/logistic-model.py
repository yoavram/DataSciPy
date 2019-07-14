def _cross_entropy(X, Y, a):
    nsamples = Y.shape[0] 
    Z = logodds(X, a)
    return -sum(
        -z*(1-y) - np.log(1+np.exp(-z)) 
        for z, y 
        in zip(Z, Y)
    ) / nsamples
        
def cross_entropy(X, Y, a):
    Z = logodds(X, a)
    logliks = -Z * (1 - Y) - np.log(1 + np.exp(-Z))
    return -logliks.mean()

def gradient_descent(X, Y, a, η=0.01):
    nsamples = Y.shape[0]
    
    Yhat = expit(logodds(X, a))
    δ = Yhat - Y
    grad = X.T @ δ / nsamples
    assert grad.shape == grad.shape
    return grad - η * grad