"""
Spliting matches into train and test sets.
Split by match not by shots.
"""

import pandas as pd

CUTOFF = "2023-07-01"  
FOLDS = 5
SEED = 2026

def splitMatches(matches):
    table = matches[["match_id", "match_date"]].copy()
    table = table.sort_values("match_id")
    table["test"] = pd.to_datetime(table["match_date"]) >= pd.to_datetime(CUTOFF)

    testMatches = table[table["test"] == True].copy()
    testMatches["fold"] = 0

    trainMatches = table[table["test"] == False].copy()
    trainMatches = trainMatches.sample(frac=1, random_state=SEED)
    folds = []
    for position in range(len(trainMatches)):
        folds.append(position % FOLDS + 1)
    trainMatches["fold"] = folds

    result = pd.concat([trainMatches, testMatches])
    return result[["match_id", "test", "fold"]]