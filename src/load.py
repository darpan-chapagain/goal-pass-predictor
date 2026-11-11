""" 
This file reads StatsBomb open data. 
Every notebooks loads the data using this file. 

"""

import os
import json
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

RAW = "../data/raw"

# Some columns record what happened after the event. So it gives the real answer away. 
# CLeaning the column to avoid data leakage.

LEAKS = [
    "shot_outcome",
    "shot_statsbomb_xg",
    "shot_end_location",
    "shot_deflected",
    "out",
    "shot_saved_to_post",
    "shot_saved_off_target",
    "pass_outcome",
    "pass_recipient",
    "pass_goal_assist",
    "pass_shot_assist",
    "pass_assisted_shot_id",
]

# ignoring warning messages from statsbombpy library

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

os.makedirs(f"{RAW}/events", exist_ok=True)
os.makedirs(f"{RAW}/frames", exist_ok=True)

def loadMatches():
    path = f"{RAW}/matches.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)

    competitions = sb.competitions()
    tables = []
    for index, row in competitions.iterrows():
        table = sb.matches(competition_id=row["competition_id"], season_id=row["season_id"])
        table["gender"] = row["competition_gender"]
        tables.append(table)
    
    matches = pd.concat(tables, ignore_index=True)
    matches["has360"] = matches["match_status_360"] == "available"
    matches.to_pickle(path)
    return matches

def loadEvents(matchId):
    path = f"{RAW}/events/{matchId}.pkl.gz"
    if os.path.exists(path):
        return pd.read_pickle(path)

    events = sb.events(match_id=matchId)
    events.to_pickle(path)
    return events

def loadFrames(matchId):
    path = f"{RAW}/frames/{matchId}.pkl.gz"
    if os.path.exists(path):
        return pd.read_pickle(path)

    frames = sb.frames(match_id=matchId)
    frames.to_pickle(path)
    return frames