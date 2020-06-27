import numpy as np


def draw(bestPath, cities):
    ax = plt.subplot(111, aspect='equal')
    ax.plot(cities[:, 0], cities[:, 1], 'x', color='blue')
    for i, city in enumerate(cities):
        ax.text(city[0], city[1], str(i))
    ax.plot(cities[bestPath, 0], cities[bestPath, 1], color='red')
    plt.show()

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import LS.LS1 as ls
    N=100
    loc = np.random.rand(N, 2)
    tour = np.random.permutation(N)

    draw(tour, loc)

    b_tour = LS_2opt(loc, tour, 1000)

    draw(b_tour, loc)
    print(ls.dist(loc, b_tour), dist(loc, b_tour))