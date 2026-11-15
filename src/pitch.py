"""
pitch geometry and plotting functions
"""
import numpy as np

GOAL_X = 120
GOAL_Y = 40
LEFT_POST_Y = 36
RIGHT_POST_Y = 44


def distanceToGoal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def goalAngle(x, y):
    toLeftPost = np.arctan2(LEFT_POST_Y - y, GOAL_X - x)
    toRightPost = np.arctan2(RIGHT_POST_Y - y, GOAL_X - x)
    return np.abs(toRightPost - toLeftPost)


def sideOfLine(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def inShotCone(shotX, shotY, playerX, playerY):
    first = sideOfLine(shotX, shotY, GOAL_X, LEFT_POST_Y, playerX, playerY)
    second = sideOfLine(GOAL_X, LEFT_POST_Y, GOAL_X, RIGHT_POST_Y, playerX, playerY)
    third = sideOfLine(GOAL_X, RIGHT_POST_Y, shotX, shotY, playerX, playerY)
    allPositive = first >= 0 and second >= 0 and third >= 0
    allNegative = first <= 0 and second <= 0 and third <= 0
    return allPositive or allNegative