#!/usr/bin/env python
# coding: utf-8
# Author: Yoav Ram (www.yoavram.com)

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

import zipfile

import numpy as np
import pandas as pd


def logodds(X, a):
    # X: (n,m), a: (m,) -> (1,m) -> (n,m)
    Z = (X * a).sum(axis=1)  # -> (n,)
    return Z


def cross_entropy(X, Y, a):
    Z = logodds(X, a)
    logliks = -Z * (1 - Y) - np.log(1 + np.exp(-Z))
    return -logliks.mean()


def expit(Z):
    return 1 / (1 + np.exp(-Z))


def gradient_descent(X, Y, a, η=0.01):
    nsamples = Y.shape[0]
    Z = logodds(X, a)
    Yhat = expit(Z)
    δ = Yhat - Y
    # X: (n,m), δ: (n,) -> (n,1) -> (n,m)
    δ = δ.reshape((nsamples, 1))
    grad = (X * δ).sum(axis=0) / nsamples # -> (m,)
    assert grad.shape == a.shape, grad.shape
    return a - η * grad


def logistic_model(X, Y, a=(1, 1, 1), iters=50000):
    a = np.array(a)
    for t in range(iters+1):
        a = gradient_descent(X, Y, a)
        if t % (iters//10) == 0:
            print("{}: loss={:.6f}, a={}".format(t, cross_entropy(X, Y, a), a))
    return a


def load_data():
    filename = 'titanic.zip'
    titanic_zip = zipfile.ZipFile(filename)
    with titanic_zip.open('data/titanic_train.csv') as f:
        df = pd.read_csv(f)


    df = df[['Sex', 'Age', 'Pclass', 'Survived']].copy() # what if I don't put .copy?
    df['Sex'] = df['Sex'] == 'female' # convert to boolean
    df['Sex'] = df['Sex'].astype(int) # then convert to int
    df = df.dropna() # remove rows with "not a number" elements

    features = ['Sex', 'Age', 'Pclass']
    X = df[features].values
    Y = df['Survived'].values
    return X, Y


if __name__ == '__main__':
    X, Y = load_data()

    a = logistic_model(X, Y)

    print("Odds-ratios:")
    print(np.exp(a))
