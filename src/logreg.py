import numpy as np

def fitScaling(X):
    """calculating  the mean and standard deviation of each feature."""
    Mean = np.mean(X, axis=0)
    Std = np.std(X, axis=0)
    Std[Std == 0] = 1
    return Mean, Std

def applyScaling(X, Mean, Std):
    """   standardizing feature using previously calculated scaling value."""

    ScaledX = (X - Mean) / Std

    return ScaledX

def sigmoid(Z):
    """converting th  model scores into probabilities between 0 and 1."""

    Z = np.clip(Z, -500, 500)

    return 1 / (1 + np.exp(-Z))

def fitLogreg(X, Y, LearningRate=0.1, Iterations=2000):
    """traing the  logistic regression using batch gradient descent."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    Rows, Columns = X.shape

    Weights = np.zeros(Columns)
    Bias = 0.0

    for _ in range(Iterations):
        Scores = X @ Weights + Bias
        Predictions = sigmoid(Scores)

        Errors = Predictions - Y

        WeightGradient = (X.T @ Errors) / Rows
        BiasGradient = np.mean(Errors)

        Weights -= LearningRate * WeightGradient
        Bias -= LearningRate * BiasGradient

    return Weights, Bias

def predictLogreg(X, Weights, Bias):
    """predicting the  class 1 probabilities using trained logistic regression."""

    X = np.asarray(X, dtype=float)

    Scores = X @ Weights + Bias
    Probabilities = sigmoid(Scores)

    return Probabilities