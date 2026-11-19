import numpy as np

def fitScaling(X):
    """c alculate the mean and standard deviation of each feature."""

    Mean = np.mean(X, axis=0)
    Std = np.std(X, axis=0)

    Std[Std == 0] = 1

    return Mean, Std
def applyScaling(X, Mean, Std):
    """standardize features using the training data's mean and standard deviation."""

    ScaledX = (X - Mean) / Std

    return ScaledX

def sigmoid(Z):
    """ cconvert model scores into probabilities between 0 and 1."""

    Z = np.clip(Z, -500, 500)

    return 1 / (1 + np.exp(-Z))

def fitLogreg(X, y, rate=0.1, steps=2000):
    """ train logistic regression using batch gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    Rows = X.shape[0]
    X = np.column_stack([np.ones(Rows), X])
    Weights = np.zeros(X.shape[1])

    for _ in range(steps):
        Scores = X @ Weights
        Predictions = sigmoid(Scores)

        Errors = Predictions - y

        Gradient = (X.T @ Errors) / Rows

        Weights -= rate * Gradient

    return Weights

def predictLogreg(weights, X):
    """Predict class 1 probabilities using trained logistic regression weights."""

    X = np.asarray(X, dtype=float)
    Rows = X.shape[0]
    X = np.column_stack([np.ones(Rows), X])
    Scores = X @ weights
    Probabilities = sigmoid(Scores)

    return Probabilities