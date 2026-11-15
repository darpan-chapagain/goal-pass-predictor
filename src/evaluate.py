"""
Every model in the project is scored.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

EDGE = 1e-6


def score(labels, predictions):
    predictions = np.clip(predictions, EDGE, 1 - EDGE)
    result = {}
    result["auc"] = roc_auc_score(labels, predictions)
    result["brier"] = brier_score_loss(labels, predictions)
    result["logloss"] = log_loss(labels, predictions)
    return result


def resultsRow(name, trainLabels, trainPredictions, validationLabels, validationPredictions, testLabels, testPredictions):
    """One row of the train / validation / test table that goes into the paper."""
    train = score(trainLabels, trainPredictions)
    validation = score(validationLabels, validationPredictions)
    test = score(testLabels, testPredictions)

    row = {"model": name}
    for split, result in [("train", train), ("validation", validation), ("test", test)]:
        for metric, value in result.items():
            # Gives column names like trainAuc, validationBrier and testLogloss.
            row[split + metric.capitalize()] = value
    return row


def calibration(labels, predictions, bins=10):
    table = pd.DataFrame({"label": labels, "prediction": predictions})
    table["bin"] = pd.qcut(table["prediction"], bins, duplicates="drop")

    grouped = table.groupby("bin", observed=True)
    result = pd.DataFrame()
    result["predicted"] = grouped["prediction"].mean()
    result["actual"] = grouped["label"].mean()
    result["rows"] = grouped["label"].size()
    return result.reset_index(drop=True)