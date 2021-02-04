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

def delete_nodes_probs(loc, tour):
    tour = np.array(tour)
    dist_ori = dist(loc, tour)
    Cs=[]
    for idx in tour:
        tour_del = tour[tour!=idx]
        del_dist = dist(loc, tour_del)
        dist_improve = max((dist_ori - del_dist),0)
        Cs.append(dist_improve)
    Cs = Cs/sum(Cs)
    return Cs


def render(loc, tour, cover_range, radius = None):
    plt.close('all')
    plt.figure(figsize=(6, 6))
    plt.scatter(loc.T[0],loc.T[1], s=20, c='r', zorder=2)
    ax = plt.gca()

    if radius is None:
        for idx in tour:
            nearest_indices = cover_idx(loc, idx, cover_range, radius)
            for nearest_idx in nearest_indices:
                pair_coor = np.concatenate((loc[idx][np.newaxis, :], loc[nearest_idx][np.newaxis, :]), 0).T
                plt.plot(pair_coor[0], pair_coor[1], linestyle='--', color='k', zorder=1, linewidth='0.5')
    else:
        for idx in tour:
            if isinstance(radius, np.ndarray):
                d = plt.Circle(loc[idx], radius[0, idx], linestyle=':', linewidth=1, color='k', fill=False)
            else:
                d = plt.Circle(loc[idx], radius, linestyle=':', linewidth=1, color='k', fill=False)
            ax.add_artist(d)
    bwith=1.8
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(top = 0.95)

    tour = np.append(tour, tour[0])
    loc_tour = loc[tour]
    plt.plot(loc_tour.T[0],loc_tour.T[1], zorder=1)
    plt.scatter(loc_tour[0, 0], loc_tour[0, 1], s=20, c='k', marker='*', zorder=3)
    plt.show()

def feasible_process(loc, tour, nodes_not_select, cover_range, radius = None):
    while not check_cover(loc, tour, cover_range, radius):
        scores, best_poss = add_node_score(loc, tour, nodes_not_select, cover_range, radius)
        if scores==[]:
            break
        add_node = nodes_not_select[np.argmin(scores)]
        add_pos = best_poss[np.argmin(scores)]
        tour = np.insert(tour, add_pos, add_node)
        nodes_not_select = np.setdiff1d(nodes_not_select, add_node)
    return tour

def add_node_score(loc, tour, nodes_not_select, cover_range, radius = None):
    uncovered_ids = get_uncovered_ids(loc, tour, cover_range, radius)
    scores=[]
    best_poss=[]
    for node in nodes_not_select:
        # if add this node. How much uncovered nodes can be covered by this node
        covered_ids = cover_idx(loc, node, cover_range, radius)
        nums_thisnode_can_cover = np.intersect1d(covered_ids, uncovered_ids).shape[0]
        if nums_thisnode_can_cover > 0:
            # find the best position to insert this node
            best_pos, minmum_increase = best_add_position(loc, tour, node)
            score = minmum_increase/nums_thisnode_can_cover**2
        else:
            score = np.inf
            best_pos = 0
        scores.append(score)
        best_poss.append(best_pos)
    return scores, best_poss

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

# def del_redundant(loc, tour, cover_range):
#     tour_ori = tour
#     for node in tour:
#         del_tour = np.setdiff1d(tour_ori, node)
#         if check_cover(loc, del_tour, cover_range):
#             tour_ori = del_tour
#     return tour_ori
def del_redundant(loc, tour, cover_range, radius = None):
    tour_ori = tour
    for node in tour_ori:
        del_tour = np.delete(tour_ori, np.where(tour_ori==node))
        if check_cover(loc, del_tour, cover_range, radius):
            tour_ori = del_tour
    return tour_ori

def mutate(loc, tour, cover_range, best_tour, best_cost, radius = None):
    r = np.random.randint(loc.shape[0])
    if r not in tour:
        # insert this node to its best place
        best_pos, _ = best_add_position(loc, tour, r)
        tour = np.insert(tour, best_pos, r)
    else:
        # remove this node, and check feasible
        # del_tour = np.setdiff1d(tour, r)
        del_tour = np.delete(tour, np.where(tour == r))
        nodes_not_select = np.setdiff1d(np.arange(loc.shape[0]), del_tour)
        tour = feasible_process(loc, del_tour, nodes_not_select, cover_range, radius)
    cost_now =  dist(loc, tour)
    if cost_now < best_cost:
        best_tour = tour
        best_cost = best_cost
    return tour, best_tour, best_cost
def LS(loc, cover_range, tour=None , print_enable=True, stop_cost=-1e6, radius = None):
    # print(np.random.seed())
    num_nodes = loc.shape[0]
    if tour is None:
        tour = init_solution(loc, cover_range, radius)
        print("Initial Solution!!")

    ids = np.arange(num_nodes)

    best_cost = dist(loc, tour)
    best_tour = tour
    iter_no_change = 0
    print("Initial city Length:", dist(loc, tour))
    for i in range(max_iters):
        nodes_not_select = np.setdiff1d(ids, tour)
        # delete nodes
        del_num = int(tour.shape[0]*del_perc)
        # del_num = 1
        del_probs = delete_nodes_probs(loc, tour)
        del_nodes = np.random.choice(tour, size=del_num, replace=False, p=del_probs)
        del_inds = [np.where(tour == node) for node in del_nodes]
        deleted_tour = np.delete(tour, del_inds)
        # feasible process
        feasible_tour = feasible_process(loc, deleted_tour, nodes_not_select, cover_range, radius)
        # del redundant nodes
        clean_tour = del_redundant(loc, feasible_tour, cover_range, radius)
        # 2opt to get the shorted path
        clean_tour = LS_2opt(loc, clean_tour, 100)

        cost_now = dist(loc, clean_tour)

        # diverse
        if if_diverse:
            if cost_now <= best_cost * (1 + a):
                tour = clean_tour
                # print("replace start-search tour, best tour/cost don't change")
                if cost_now < best_cost:
                    best_tour = tour
                    best_cost = cost_now
                    iter_no_change = 0
                    # print("replace All")
            else:
                tour = best_tour
                iter_no_change = iter_no_change + 1
        else:
            if cost_now < best_cost:
                # print("replace All")
                tour = clean_tour
                best_tour = tour
                best_cost = cost_now
            else:
                iter_no_change = iter_no_change + 1

        # mutate
        if if_mutate:
            if iter_no_change > mutate_iters:
                tour, best_tour, best_cost = mutate(loc, tour, cover_range, best_tour, best_cost, radius)
        # if iter_no_change > loc.shape[0]:
        if iter_no_change > 50:
            break
        # log
        if print_enable:
            if i % 40 == 0:
                print("Iteration %d, Current distance: %2.3f" % (i, best_cost))
    return best_tour

# del_perc = 0.1
# max_iters = 400
# mutate_iters = 15
# if_diverse = False
# if_mutate = False
del_perc = 0.1
max_iters = 500
mutate_iters = 15
if_diverse = False
if_mutate = False
a = 0.1
if __name__ == '__main__':
    # np.random.seed(123)
    loc=np.random.rand(50, 2)
    import time
    t1=time.time()
    tour = LS(loc, 7)
    print(time.time()-t1)
    # tour = init_solution(loc, 7)
    render(loc,tour,7)
    print("Final cost:",dist(loc,tour))
    # print(delete_nodes_probs(loc, tour))