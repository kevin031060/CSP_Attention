import numpy as np
import torch
from torch.utils.data import Dataset
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TSPLIB_dataset(Dataset):

    def __init__(self, tsp_name, range=7):
        super(TSPLIB_dataset, self).__init__()

        tsp_name = os.path.join("TSPLIB", '%s.tsp'%tsp_name)

        x = np.loadtxt(tsp_name, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float, comments="EOF")
        self.original_data = torch.from_numpy(x).float()
        num_nodes = x.shape[0]
        # normalize
        x = x / (np.max(x))
        x = x.T
        # add batch axis
        x = x[np.newaxis, :]

        self.dataset = torch.from_numpy(x).float()
        self.range = range
        self.num_nodes = num_nodes
        self.size = 1
        self.dynamic = torch.ones(1, 1, num_nodes).float()
        self.dynamic_updation = torch.arange(range + 1).flip(0).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])

    def back_tour(self, tour_indices):
        tour_indices = tour_indices.repeat(2,1).transpose(1,0)
        tour = torch.gather(self.original_data, 0, tour_indices)
        tour = torch.cat((tour, tour[0].unsqueeze(0)))
        tour_len = torch.sqrt(torch.sum(torch.pow(tour[1:]-tour[:-1], 2),dim=1))
        total_len = tour_len.sum()
        return total_len

    def update_dynamic(self, dynamic, chosen_idx, nearest_index):
        batch_size, _, _ = dynamic.size()

        dynamic_ = dynamic.clone()
        dynamic_ = dynamic_.squeeze(1).cpu()
        dynamic_ = dynamic_.mul(self.range)
        for i in range(batch_size):
            dynamic_[i,nearest_index[i]] = torch.clamp(dynamic_[i,nearest_index[i]]-self.dynamic_updation, min=0.1)
        dynamic_ = dynamic_/self.range
        return torch.tensor(dynamic_.data, device = dynamic.device).unsqueeze(1)

    def update_mask(self, mask, static, chosen_idx):
        batch_size, input_size, sequence_size = static.size()
        # static:(256,2,20) idx:(256).
        # for coor, idx in zip(static, chosen_idx)--> coor:(2,20), coor[:,idx] is the coordinate of the chosen city.
        # coor[:,idx].repeat((sequence_size,1)): (20,2)----> then .transpose(1,0)---> (2,20). Then it has the same dim with coor
        # torch.sum(torch.pow(coor- ... )) calculate the distances between the chosen city and all cities
        # distances_batch: list [] of 256. each element is a [20] size tensor, representing the distances.

        distances_batch = [
            torch.sum(torch.pow(coor - coor[:, idx].repeat((sequence_size, 1)).transpose(1, 0), 2), dim=0) for coor, idx
            in zip(static, chosen_idx)]
        # for each batch, get the nearest 7 city. get its index
        # nearest_index: list [] of 256. each element is a [7] size array, representing the index of nearest 7 city
        nearest_index = [distances.argsort()[0:self.range + 1] for distances in distances_batch]
        # scatter_(1, torch.LongTensor(nearest_index), 0)
        # 1 means for each row i. we replace the elements on nearest_index(i,:) to 0.
        for i in range(batch_size):
            mask[i, nearest_index[i]] = 0
        return mask.detach(), nearest_index


def reward(static, tour_indices, done_steps):
    # tour_indices: (256,6)
    rewards = [cal_total_distance(static_, tour_idx[:done_step]) for static_, tour_idx, done_step in zip(static, tour_indices, done_steps)]
    return rewards

def cal_total_distance(static_, tour_idx):
    # tour_idx (6) static_: (2,20)
    idx = tour_idx.repeat(2, 1) # (2,6)
    tour = torch.gather(static_.data, 1, idx).permute(1, 0) # (6,2)
    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[0].unsqueeze(0)), dim=0)
    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:-1] - y[1:], 2), dim=1))
    total_len = tour_len.sum(0).detach()
    return total_len.unsqueeze(0)

def cal_nearest_indices(static_, idx, nearest_num=7):
    _, sequence_size = static_.size()
    distances = torch.sum(torch.pow(static_ - static_[:,idx].repeat(sequence_size,1).transpose(1,0), 2), dim=0)
    # for each batch, get the nearest 7 city. get its index
    # nearest_index: list [] of 256. each element is a [7] size array, representing the index of nearest 7 city
    nearest_index = distances.argsort()[1:nearest_num+1]
    return nearest_index


def render(static, tour_indices, done_steps, save_path, test = False):
    """Plots the found tours."""
    if not test:
        matplotlib.use('Agg')
    matplotlib.use('Agg')

    plt.close('all')
    static = static.cpu()
    tour_indices = tour_indices.cpu()
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i, :done_steps[i]]
        static_ = static[i].data

        for idx_ in idx:
            nearest_indices = cal_nearest_indices(static_, idx_)
            for nearest_idx in nearest_indices:
                pair_coor = torch.cat((static_[:,idx_].unsqueeze(0),static_[:,nearest_idx].unsqueeze(0))).transpose(1,0).numpy()
                ax.plot(pair_coor[0], pair_coor[1], linestyle='--', color='k',zorder=1,  linewidth='0.5')


        # End tour at the starting index
        idx = torch.cat((idx, idx[0].unsqueeze(0)))
        idx = idx.repeat(2, 1)  # (2,6)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(static_[0], static_[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if test:
        plt.show()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
