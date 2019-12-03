import numpy as np


def dist(loc, tour):
    loc = loc[tour]
    return np.sqrt(np.square(loc[1:]-loc[:-1]).sum(-1)).sum() + np.sqrt(np.square(loc[0]-loc[-1]).sum())

# path1长度比path2短则返回true
def pathCompare(path1, path2, loc):
    if dist(loc, path1) <= dist(loc, path2):
        return True
    return False


def generateRandomPath(bestPath):
    a, b = np.random.choice(np.arange(len(bestPath)),2)
    if a > b:
        return b, a, bestPath[b:a + 1]
    else:
        return a, b, bestPath[a:b + 1]


def reversePath(path):
    rePath = path.copy()
    rePath[1:-1] = rePath[-2:0:-1]
    return rePath


def LS_2opt(loc, bestPath, MAXCOUNT):
    count = 0
    while count < MAXCOUNT:

        start, end, path = generateRandomPath(bestPath)
        # print(path)
        rePath = reversePath(path)
        # print(rePath)
        if pathCompare(path, rePath, loc):
            count += 1
            continue
        else:
            count = 0
            bestPath[start:end + 1] = rePath
    return bestPath