import numpy as np
import torch

def dis_matrix(test_loader, x=None):
    if x is not None:
        loc = x
    else:
        iter_data = iter(test_loader)
        static, dynamic, x0 = iter_data.next()
        loc = static.squeeze(0)

    _, l = loc.size()

    matrix = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i != j:
                matrix[i,j] = torch.sqrt(torch.sum(torch.pow(loc[:, i] - loc[:, j], 2))).detach()
    if x is None:
        matrix = matrix * 1000
    return matrix, loc.transpose(1,0)

def write_cplex(matrix, loc, cover_range=7):
    size = matrix.shape[0]

    nearest_index = [distances.argsort()[0:cover_range + 1] for distances in matrix]
    cover_matrix = np.zeros(matrix.shape)
    for i in range(size):
        cover_matrix[i, nearest_index[i]] = 1
    cover_matrix = cover_matrix.T

    with open("data.dat", "w") as f:  # 设置文件对象
        f.writelines("n=%d;"%size + '\r\n')
        f.writelines("dist=[" + '\r\n')
        for i in range(size):
            f.write("[")
            for j in range(size):
                f.write("%2.3f "%matrix[i,j])
            f.writelines("]" + '\r\n')
        f.writelines("];" + '\r\n')
        f.writelines("cover=[" + '\r\n')
        for i in range(size):
            f.write("[")
            for j in range(size):
                f.write("%d "%cover_matrix[i,j])
            f.writelines("]" + '\r\n')

        f.writelines("];")
    return cover_matrix

def write_yalmip(matrix, loc, cover_range=7):
    size = matrix.shape[0]

    nearest_index = [distances.argsort()[0:cover_range + 1] for distances in matrix]
    cover_matrix = np.zeros(matrix.shape)
    for i in range(size):
        cover_matrix[i, nearest_index[i]] = 1
    cover_matrix = cover_matrix.T

    with open("dist_csp.txt", "w") as f:  # 设置文件对象

        for i in range(size):
            for j in range(size):
                f.write("%2.3f "%matrix[i,j])
            f.writelines('\r\n')


    with open("cover.txt", "w") as f:  # 设置文件对象
        for i in range(size):
            for j in range(size):
                f.write("%d "%cover_matrix[i,j])
            f.writelines('\r\n')
    return cover_matrix

def write_cplex2(matrix, loc, cover_range=7):
    size = matrix.shape[0]

    nearest_index = [distances.argsort()[0:cover_range + 1] for distances in matrix]
    cover_matrix = np.zeros(matrix.shape)
    for i in range(size):
        cover_matrix[i, nearest_index[i]] = 1
    cover_matrix = cover_matrix.T

    with open("data.dat", "w") as f:  # 设置文件对象
        f.writelines("n=%d;"%size + '\r\n')
        f.writelines("subtours = {};"+ '\r\n')
        f.writelines("dist=[" + '\r\n')
        for i in range(size):
            for j in range(i + 1, size):
                f.write("%2.3f "%matrix[i,j] + '\r\n')
        f.writelines("];" + '\r\n')


        f.writelines("cover=[" + '\r\n')
        for i in range(size):
            f.write("[")
            for j in range(size):
                f.write("%d "%cover_matrix[i,j])
            f.writelines("]" + '\r\n')

        f.writelines("];")

    with open("loc.txt", "w") as f:  # 设置文件对象
        for i in range(size):
            for j in range(2):
                f.write("%2.3f " % loc[i, j])
            f.write('\r\n')
    with open("cover.txt", "w") as f:  # 设置文件对象
        for i in range(size):
            for j in range(size):
                f.write("%d "%cover_matrix.T[i,j])
            f.write('\r\n')
    return cover_matrix