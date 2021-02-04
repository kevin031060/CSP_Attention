import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LS.LS_2opt import LS_2opt



def init_solution(loc, cover_range, radius = None):
    node_num, _ = loc.shape
    ids = np.arange(node_num)
    uncovered_indices = ids
    num_uncovered = node_num
    tour=[]

    while num_uncovered>0:
        n_selected = np.random.choice(uncovered_indices)
        tour.append(n_selected)
        cover_indices = cover_idx(loc, n_selected, cover_range, radius)
        uncovered_indices = np.array([idx for idx in uncovered_indices if idx not in cover_indices])
        num_uncovered = uncovered_indices.shape[0]

    # print('Init_solution:',tour)
    # print('Assert covered check:',check_cover(loc, tour, cover_range))
    return np.array(tour)

def cover_idx(loc, chosen_idx, cover_range, radius = None):
    dists = np.sqrt(np.square(loc-loc[chosen_idx]).sum(-1))
    if radius is None:
        if isinstance(cover_range, int):
            return dists.argsort()[:cover_range+1]
        else:
            return dists.argsort()[:cover_range[chosen_idx]+1]
    else:
        if isinstance(radius, np.ndarray):
            return dists.argsort()[np.sort(dists) <= radius[0, chosen_idx]]
        else:
            return dists.argsort()[np.sort(dists) <= radius]
def check_cover(loc, tour, cover_range, radius = None):
    mask = np.zeros(loc.shape[0])
    for idx in tour:
        cover_indices = cover_idx(loc, idx, cover_range, radius)
        mask[cover_indices]=1
    return mask.all()

def get_uncovered_ids(loc, tour, cover_range, radius = None):
    mask = np.zeros(loc.shape[0])
    for idx in tour:
        cover_indices = cover_idx(loc, idx, cover_range, radius)
        mask[cover_indices]=1
    return np.where(mask==0)

def dist(loc, tour):
    loc = loc[tour]
    return np.sqrt(np.square(loc[1:]-loc[:-1]).sum(-1)).sum() + np.sqrt(np.square(loc[0]-loc[-1]).sum())



def render(loc, tour, cover_range, radius = None):
    plt.close('all')
    plt.scatter(loc.T[0],loc.T[1], s=4, c='r', zorder=2)
    ax = plt.gca()
    if radius is None:
        for idx in tour:
            nearest_indices = cover_idx(loc, idx, cover_range, radius)
            for nearest_idx in nearest_indices:
                pair_coor = np.concatenate((loc[idx][np.newaxis, :], loc[nearest_idx][np.newaxis, :]), 0).T
                plt.plot(pair_coor[0], pair_coor[1], linestyle='--', color='k', zorder=1, linewidth='0.5')
    else:
        for idx in tour:
            d = plt.Circle(loc[idx], radius, fill=False)
            ax.add_artist(d)

    tour = np.append(tour, tour[0])
    loc_tour = loc[tour]
    plt.plot(loc_tour.T[0],loc_tour.T[1], zorder=1)
    plt.scatter(loc_tour[0, 0], loc_tour[0, 1], s=20, c='k', marker='*', zorder=3)
    plt.show()



def best_add_position(loc, tour, node):
    dist_ori = dist(loc, tour)
    # Look up all insert position, calculate the increase of distance
    best_pos = 0
    minmum_increase = np.inf
    for pos in range(tour.shape[0]):
        tour_insert = np.insert(tour, pos, node)
        dist_increase = dist(loc, tour_insert) - dist_ori
        if dist_increase<minmum_increase:
            best_pos = pos
            minmum_increase = dist_increase
    return best_pos, minmum_increase

def del_redundant(loc, tour, cover_range, radius = None):
    tour_ori = tour
    for node in tour_ori:
        del_tour = np.delete(tour_ori, np.where(tour_ori==node))
        if check_cover(loc, del_tour, cover_range, radius):
            tour_ori = del_tour
    return tour_ori




def subsitute_by_neighbor(loc, del_tour, del_pos, del_node, ori_cost, cover_range, radius = None):
    neighbours = cover_idx(loc, del_node, loc.shape[0]-1, radius)[1:]
    neighbours = [i for i in neighbours if i not in del_tour]
    neighbours = neighbours[:min(T, len(neighbours))]
    # print(del_node, del_tour, neighbours)
    best_cost = np.inf
    best_tour = None
    for node in neighbours:
        tour = np.insert(del_tour, del_pos, node)
        cost_now = dist(loc, tour)
        if cost_now<ori_cost:
            if check_cover(loc, tour, cover_range, radius):
                if cost_now < best_cost:
                    best_cost = cost_now
                    best_tour = tour

    return best_tour

def perturbation_process(loc, tour, ids):
    for i in range(K):
        nodes_not_selected = np.setdiff1d(ids, tour)
        node = np.random.choice(nodes_not_selected)
        best_pos, _ = best_add_position(loc, tour, node)
        tour = np.insert(tour, best_pos, node)
    return tour

def improve_process(loc, tour, cover_range, radius = None):
    improve = False
    ori_cost = dist(loc, tour)
    Ns = tour.shape[0]
    del_pos = 0
    while del_pos < Ns:
        del_node = tour[del_pos]
        del_tour = np.delete(tour, del_pos)
        if check_cover(loc, del_tour, cover_range, radius):
            improve = True
            tour = del_tour
        else:
            subsituted_tour = subsitute_by_neighbor(loc, del_tour, del_pos, del_node, ori_cost, cover_range, radius)
            if subsituted_tour is not None:
                tour = subsituted_tour
                improve = True
            del_pos = del_pos + 1
        Ns = tour.shape[0]

    return improve, tour


def LS(loc, cover_range, tour=None, print_if=True, stop_cost=-1e6, radius = None):
    num_nodes = loc.shape[0]
    if tour is None:
        tour = init_solution(loc, cover_range, radius)
    ids = np.arange(num_nodes)
    # render(loc, tour, cover_range)
    best_tour = tour
    best_cost = dist(loc, tour)
    iter_no_change_outer = 0
    for i in range(max_iters):
        bestimprove = False
        iter_no_change = 0
        for j in range(J):
            improve = True
            while improve is True:
                improve, tour = improve_process(loc, tour, cover_range, radius)
            tour = LS_2opt(loc, tour, 100)
            cost_now = dist(loc, tour)
            if cost_now < best_cost:
                best_tour = tour
                best_cost = cost_now
                bestimprove = True
                iter_no_change = 0
            else:
                tour = best_tour
                iter_no_change = iter_no_change + 1
            tour = perturbation_process(loc, tour, ids)
            if iter_no_change > 50:
                break
        if bestimprove is True:
            iter_no_change_outer = 0
            best_tour = LS_2opt(loc, best_tour, 200)
            cost_now = dist(loc, best_tour)
            tour = best_tour
            best_cost = cost_now
        iter_no_change_outer = iter_no_change_outer + 1
        if iter_no_change_outer > 3:
            break
        if print_if:
            print('Iter,', i, ' cost:', best_cost)
        if best_cost <= stop_cost:
            return best_tour

    return best_tour


max_iters = 25

J=150
T=10
K=10

if __name__ == '__main__':
    # np.random.seed(123)
    loc=np.random.rand(20, 2)
    import time
    t1=time.time()
    tour = LS(loc, 7)
    print(time.time()-t1)
    # tour = init_solution(loc, 7)
    render(loc,tour,7)
    print("Final cost:",dist(loc,tour))
    # print(delete_nodes_probs(loc, tour))