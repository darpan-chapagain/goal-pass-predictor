import numpy as np
def fitScaling(X):
    """Calculate the mean and standard deviation of each feature."""
    mean = np.mean(X, axis=0)
    spread = np.std(X, axis=0)
    spread[spread == 0] = 1
    return mean, spread

def applyScaling(X, mean, spread):
    """Standardize features using the training data's mean and standard deviation."""

    scaledX = (X - mean) / spread

    return scaledX

def sigmoid(z):
    """Convert model scores into probabilities between 0 and 1."""

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))

def fitLogreg(X, y, rate=0.1, steps=2000, l2=0.0, tolerance=None, returnSteps=False):
    """Train logistic regression using batch gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    rows = X.shape[0]
    X = np.column_stack([np.ones(rows), X])
    weights = np.zeros(X.shape[1])
    previousCost = None
    stepsUsed = 0

    for _ in range(steps):
        stepsUsed += 1
        scores = X @ weights
        predictions = sigmoid(scores)
        errors = predictions - y
        gradient = (X.T @ errors) / rows
        gradient[1:] += (l2 / rows) * weights[1:]
        weights -= rate * gradient

        if tolerance is not None:
            scores = X @ weights
            cost = np.mean(np.logaddexp(0, scores) - y * scores)
            cost += (l2 / (2 * rows)) * np.sum(weights[1:] ** 2)

            if previousCost is not None:
                if abs(previousCost - cost) < tolerance:
                    break

            previousCost = cost
    if returnSteps:
        return weights, stepsUsed

    return weights

def predictLogreg(weights, X):
    """Predict class 1 probabilities using trained logistic regression weights."""

    X = np.asarray(X, dtype=float)
    rows = X.shape[0]
    X = np.column_stack([np.ones(rows), X])
    scores = X @ weights
    probabilities = sigmoid(scores)

    return probabilities