import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    # mask_cover
    mask_cover: torch.Tensor
    cover_range: torch.Tensor
    dynamic: torch.Tensor
    dynamic_updation: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                mask_cover=self.mask_cover[key],
                dynamic=self.dynamic[key],
                dynamic_updation=self.dynamic_updation[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateCSP, self).__getitem__(key)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        loc = input['loc']
        cover_range = input['cover_range']
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        return StateCSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            mask_cover=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            cover_range=cover_range[0],
            dynamic=torch.ones(batch_size, 1, n_loc, dtype=torch.float, device=loc.device),
            # dynamic_updation=torch.log(torch.arange(cover_range[0], device=loc.device).float()+1.0).
            #                      expand(batch_size,1,-1)/torch.log(cover_range[0].float()),
            dynamic_updation=torch.arange(cover_range[0], device=loc.device).float().expand(batch_size, 1, -1) /
                             cover_range[0],

            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        # mask covered cities
        batch_size, sequence_size, _ = self.loc.size()
        dists = (self.loc[self.ids.squeeze(-1)]-cur_coord).norm(p=2, dim=-1)
        nearest_idx = dists.argsort()[:,:self.cover_range+1]
        mask_cover = self.mask_cover.scatter(-1, nearest_idx.unsqueeze(1), 1)

        # update dynamic, give covered cities lower values
        nearest_idx = nearest_idx[:,1:]
        dynamic = self.dynamic.scatter(-1, nearest_idx.unsqueeze(1),
                                       self.dynamic.gather(-1, nearest_idx.unsqueeze(1)).mul(self.dynamic_updation))
        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, mask_cover=mask_cover,dynamic=dynamic,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.mask_cover.all()
    def get_finished(self):
        return self.mask_cover.sum(-1) == self.mask_cover.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_dynamic(self):
        return self.dynamic

    def get_mask(self):
        return self.visited

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
