import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def softmax(x):
    expx = np.exp(x - x.max(axis=1, keepdims=True))
    return expx / expx.sum(axis=1, keepdims=True)


def predict(W, X):
    if X.ndim == 1: 
        # if we got a single sample, so X is 1D we convert X to 2D with just one row
        # to be consistent with cases in which we get multiple samples (rows)
        X = X[np.newaxis, :]
    Z = X @ W
    Yhat = softmax(Z)
    return Yhat

def sklearn_softmax_model(X, Y):
    model = LogisticRegression(C=1e12, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
    model.fit(X, Y.argmax(axis=1))
    return model