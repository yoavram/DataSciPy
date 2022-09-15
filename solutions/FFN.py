import numpy as np
import scipy.stats

## ReLU

def ReLU(X):
    return np.maximum(X, 0.0)

def dReLU(X):
    return (X > 0).astype(float)

## Feed Forward with loop

def feed_forward(Ws, X, keep_prob=1):
    layers = [X] # input layer
    for i, W in enumerate(Ws):
        X = X @ W 
        layers.append(X)
        if i < len(Ws) - 1:
            # hidden layer
            if keep_prob < 1:
                X = drop(X, keep_prob=keep_prob)
            X = ReLU(X) 
        else:
            # readout layer
            X = softmax(X) 
        layers.append(X)
    return layers

## Back propagation with loop

def back_propagation(Ws, X, Y, keep_prob=1):
    layers = feed_forward(Ws, X, keep_prob=keep_prob) # X1, Z1, X2, Z2, Yhat
    dJdWs = []
    
    for i in range(len(Ws)):
        Z = layers.pop() # remove last layer from list
        if i == 0:
            # readout layer, Z=Yhat
            δ = Z - Y
            layers.pop() # remove last layer from list
        else:
            # hidden layers, Z = X @ W
            W = Ws[-i]
            δ = (δ @ W.T) * dReLU(Z) # δ = δ * W * ReLU(Z)
        X = layers.pop() # remove last layer from list
        dJdW = X.T @ δ # dC/dW = δ * X
        dJdWs.append(dJdW)
    
    # reverse list of gradients - it is currently from last to first
    dJdWs.reverse() 
    # sanity checks
    assert len(dJdWs) == len(Ws), (len(dJdWs), len(Ws))
    for dJdW, W in zip(dJdWs, Ws):
        assert dJdW.shape == W.shape, (dJdW.shape, W.shape)
    return dJdWs
