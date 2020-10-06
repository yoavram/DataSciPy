# Generative model assignment

def feed_forward(Ws, X, keep_prob=1):
    layers = [X]
    for i, W in enumerate(Ws):
        X = X @ W
        layers.append(X)
        if i < len(Ws) - 1:
            # hidden layer
            if keep_prob < 1:
                X = drop(X, keep_prob=keep_prob)
            X = np.tanh(X)
        else:
            # readout layer
            X = softmax(X)
        layers.append(X)
    return layers

def back_propagation(Ws, X, Y, keep_prob=1):
    layers = feed_forward(Ws, X, keep_prob=keep_prob)

    gradients = []
    # readout layer
    Yhat = layers.pop() # softmax layer
    layers.pop() # linear layer
    δ = Yhat - Y # derivative of loss wrt softmax layer
    δ = δ[:,np.newaxis,:]
    X = layers.pop() # previous layer, after Relu
    X = X[:, :, np.newaxis] 
    dW = (δ * X).mean(axis=0) # 
    gradients.append(dW)

    # hidden layers
    for W in reversed(Ws[1:]):
        Z = layers.pop()
        dZ = dtanh(Z)
        reverseW = W.T[np.newaxis, :, :]
        δ = (δ @ reverseW).squeeze() * dZ # squeeze removes dimensions with size 1
        δ = δ[:, np.newaxis, :]
        X = layers.pop()
        X = X[:, :, np.newaxis]
        dW = (δ * X).mean(axis=0)
        gradients.append(dW)
 
    gradients = gradients[::-1] # reverse gradients list
    # sanity checks
    assert len(gradients) == len(Ws), (len(gradients), len(Ws))
    for dW, W in zip(gradients, Ws):
        assert dW.shape == W.shape, (dW.shape, W.shape)
    return gradients

def feed_backward(Ws, X):
    for i, W in enumerate(reversed(Ws)):
        if i > 0:
            # hidden layer
            X = np.arctanh(X) # X = tanh(X)
        else:
            # readout layer
            X = -np.log(X) # X = softmax(X)
            X /= X.sum() # normalize
        X = X @ W.T # X = X @ W
    return X
