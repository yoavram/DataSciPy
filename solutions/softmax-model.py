from sklearn.linear_model import LogisticRegression

def sklearn_softmax_model(X, Y):
    model = LogisticRegression(C=1e12, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
    model.fit(X, Y.argmax(axis=1))
    return model