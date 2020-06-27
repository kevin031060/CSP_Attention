from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.csp.state_csp import StateCSP
from utils.beam_search import beam_search


class CSP(object):

    NAME = 'csp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #     pi.data.sort(1)[0]
        # ).all(), "Invalid tour"
        loc = dataset['loc']
        batch_size,_,_ = loc.size()

        lengths = [cal_dist(coor.gather(0, tour[tour>-1].expand(2,-1).transpose(1,0))).unsqueeze(0) for coor, tour in zip(loc, pi)]

        center_distance = [cal_center_dist(coor.gather(0, tour[tour > -1].expand(2, -1).transpose(1, 0))).unsqueeze(0) for coor, tour
                   in zip(loc, pi)]

        # Gather dataset in order of tour
        lengths = torch.cat(lengths)
        center_distance = torch.cat(center_distance)
        nums = [tour[tour>-1].size(-1) for tour in pi]

        # print(center_distance)
        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        # return torch.tensor(nums, device=pi.device).float()/10+lengths, None

        # if (torch.rand(1)<0.1):
        #     print(torch.mean(lengths))
        return lengths, None



    @staticmethod
    def make_dataset(*args, **kwargs):
        return CSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

def cal_center_dist(loc):

    center = torch.ones((loc.size(0), loc.size(1)), device=loc.device) * 0.5
    return (loc - center).norm(p=2, dim=-1).sum()

def cal_dist(ordered_loc):
    return (ordered_loc[1:, :] - ordered_loc[:-1, :]).norm(p=2, dim=-1).sum() + (
                    ordered_loc[0, :] - ordered_loc[-1, :]).norm(p=2, dim=-1)

class CSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=500000, offset=0, cover_range=7, distribution=None):
        super(CSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                {
                    'loc':torch.FloatTensor(size, 2).uniform_(0, 1),
                    'cover_range':cover_range
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)
        self.cover_range = cover_range

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
