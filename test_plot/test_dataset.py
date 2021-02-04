from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Test_CSPDataset(Dataset):

    def __init__(self, size=50, num_samples=500000, cover_range=7, radius = 0.3, seed = None, tsp_name=None, sample_mode=False, test_instance=None, varible_NC=False):
        super(Test_CSPDataset, self).__init__()

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if radius is None:
            if sample_mode:
                loc = torch.rand(size, 2)
                self.data = [
                    {
                        'loc': loc,
                        'cover_range': cover_range
                    }
                    for i in range(num_samples)
                ]
            else:
                self.data = [
                    {
                        'loc': torch.rand(size, 2),
                        'cover_range': cover_range
                    }
                    for i in range(num_samples)
                ]
        else:
            if sample_mode:
                loc = torch.rand(size, 2)
                self.data = [
                    {
                        'loc': loc,
                        'cover_range': cover_range,
                        'radius': radius
                    }
                    for i in range(num_samples)
                ]
            else:
                if tsp_name is None:
                    if test_instance is None:
                        self.data = [
                            {
                                'loc': torch.rand(size, 2),
                                'cover_range': cover_range
                            }
                            for i in range(num_samples)
                        ]
                    else:
                        self.data = [
                            {
                                'loc': torch.Tensor(test_instance),
                                'cover_range': cover_range
                            }
                            for i in range(num_samples)
                        ]
                else:
                    dataset = tsplib_dataset(tsp_name)
                    self.data = [
                        {
                            'loc': dataset[i],
                            'cover_range': cover_range
                        }
                        for i in range(num_samples)
                    ]
        if varible_NC:
            loc = torch.rand(size, 2)
            cover_range = torch.randint(2, 16, (size, ))
            self.data = [
                {
                    'loc': loc,
                    'cover_range': cover_range,
                }
                for i in range(num_samples)
            ]
        self.size = len(self.data)
        self.cover_range = cover_range

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def tour_len_dist(batch, pi):
    loc = batch['loc']
    batch_size, _, _ = loc.size()

    dists = [cal_dist(coor.gather(0, tour[tour > -1].expand(2, -1).transpose(1, 0))).unsqueeze(0) for coor, tour in
               zip(loc, pi)]
    # Gather dataset in order of tour
    dists = torch.cat(dists)
    lens = [tour[tour > -1].size(-1) for tour in pi]
    return dists, lens

def cal_dist(ordered_loc):
    return (ordered_loc[1:,:]-ordered_loc[:-1,:]).norm(p=2,dim=-1).sum() + (ordered_loc[0,:]-ordered_loc[-1,:]).norm(p=2,dim=-1)
def clean_tour(tour):
    return tour[tour > -1]

def tsplib_dataset(tsp_name):
    tsp_name = os.path.join("TSPLIB", '%s.tsp' % tsp_name)

    x = np.loadtxt(tsp_name, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float, comments="EOF")
    original_data = torch.from_numpy(x).float()
    num_nodes = x.shape[0]
    # normalize
    x = x / (np.max(x))
    x = x.T


    dataset = torch.from_numpy(x).float().transpose(0,1)
    return dataset, num_nodes, original_data

def render(static, tour_indices, save_path, test = False):
    """Plots the found tours."""
    # if not test:
    #     matplotlib.use('Agg')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    plt.close('all')
    if isinstance(static, torch.Tensor):
        static = static.cpu()
    if isinstance(tour_indices, torch.Tensor):
        tour_indices = tour_indices.cpu()

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i, :]
        idx = idx[idx>-1]
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

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    print(save_path)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)

def cal_nearest_indices(static_, idx, nearest_num=7):
    _, sequence_size = static_.size()
    distances = torch.sum(torch.pow(static_ - static_[:,idx].repeat(sequence_size,1).transpose(1,0), 2), dim=0)
    # for each batch, get the nearest 7 city. get its index
    # nearest_index: list [] of 256. each element is a [7] size array, representing the index of nearest 7 city
    nearest_index = distances.argsort()[1:nearest_num+1]
    return nearest_index